import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.config import get_cfg
import logging
import os
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter
import torch, torchvision
from detectron2.data.datasets import register_coco_instances,load_coco_json

register_coco_instances("dataset_train_synthetic", {}, "Bellomo_Dataset_MDA/synthetic/Object_annotations/Training_annotations.json", "Bellomo_Dataset_MDA/synthetic/images")

register_coco_instances("dataset_train_real", {}, "Bellomo_Dataset_MDA/real_hololens/training/training_set.json", "Bellomo_Dataset_MDA/real_hololens/training")
register_coco_instances("dataset_train_real2", {},"Bellomo_Dataset_MDA/real_gopro/Training/training_set.json","Bellomo_Dataset_MDA/real_gopro/Training")

register_coco_instances("dataset_test_real", {}, "Bellomo_Dataset_MDA/real_hololens/test/test_set.json", "Bellomo_Dataset_MDA/real_hololens/test")
register_coco_instances("dataset_test_real2", {},"Bellomo_Dataset_MDA/real_gopro/Test/test_set.json","Bellomo_Dataset_MDA/real_gopro/Test")

logger = logging.getLogger("detectron2")

def do_train(cfg_source, cfg_target, cfg_target2,  model, resume = False):

    model.train()
    print(model)

    optimizer = build_optimizer(cfg_source, model)
    scheduler = build_lr_scheduler(cfg_source, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg_source.OUTPUT_DIR, optimizer = optimizer, scheduler = scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg_source.MODEL.WEIGHTS, resume = resume).get("iteration", -1) + 1
    )
    max_iter = cfg_source.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg_source.SOLVER.CHECKPOINT_PERIOD, max_iter = max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg_source.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg_source.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    i = 1
    current_epoch = 0
    max_epoch = 40 #max iteration / min (data_len(source, target_1, target_2))
    data_len = 1502

    data_loader_source = build_detection_train_loader(cfg_source)
    data_loader_target = build_detection_train_loader(cfg_target)
    data_loader_target2 = build_detection_train_loader(cfg_target2)

    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for data_source, data_target, data_target2, iteration in zip(data_loader_source, data_loader_target, data_loader_target2, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            if (iteration % data_len) == 0:
                current_epoch += 1
                i = 1

            p = float( i + current_epoch * data_len) / max_epoch / data_len
            alpha = 2. / ( 1. + np.exp( -10 * p)) - 1
            i += 1
        
            if alpha > 0.5:
                alpha = 0.5

            loss_dict = model(data_source, "source", alpha)
            loss_dict_target = model(data_target, "target_1", alpha)
            loss_dict_target2 = model(data_target2, "target_2", alpha)

            loss_dict["loss_r3"] += loss_dict_target["loss_r3"]
            loss_dict["loss_r3"] += loss_dict_target2["loss_r3"]
            loss_dict["loss_r3"] *= 0.2
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

cfg_source = get_cfg()
cfg_source.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg_source.DATASETS.TRAIN = ("dataset_train_synthetic",)
cfg_source.DATALOADER.NUM_WORKERS = 0
cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg_source.SOLVER.IMS_PER_BATCH = 4
cfg_source.SOLVER.BASE_LR = 0.0002
cfg_source.SOLVER.WEIGHT_DECAY = 0.001
cfg_source.SOLVER.MAX_ITER = 60000
cfg_source.SOLVER.STEPS = (30000,)
cfg_source.INPUT.MIN_SIZE_TRAIN = (0,)
cfg_source.INPUT.MIN_SIZE_TEST = 0
os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
cfg_source.MODEL.RETINANET.NUM_CLASSES = 16
model = build_model(cfg_source)

cfg_target = get_cfg()
cfg_target.DATASETS.TRAIN = ("dataset_train_real",)
cfg_target.INPUT.MIN_SIZE_TRAIN = (0,)
cfg_target.DATALOADER.NUM_WORKERS = 0
cfg_target.SOLVER.IMS_PER_BATCH = 2

cfg_target2 = get_cfg()
cfg_target2.DATASETS.TRAIN = ("dataset_train_real2",)
cfg_target2.INPUT.MIN_SIZE_TRAIN = (0,)
cfg_target2.DATALOADER.NUM_WORKERS = 0
cfg_target2.SOLVER.IMS_PER_BATCH = 2

do_train(cfg_source, cfg_target, cfg_target2, model)

#test Hololens
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
evaluator = COCOEvaluator("dataset_test_real", cfg_source, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg_source, "dataset_test_real")
inference_on_dataset(model, val_loader, evaluator)

#test GoPro
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
evaluator = COCOEvaluator("dataset_test_real2", cfg_source, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg_source, "dataset_test_real2")
inference_on_dataset(model, val_loader, evaluator)
