# Detectron2 implementation of STMDA-RetinaNet
This is the implementation of our Computer Vision and Image Understanding 2022 work 'A Multi Camera Unsupervised Domain Adaptation Pipeline for Object Detection in Cultural Sites through Adversarial Learning and Self-Training'. The aim is to reduce the gap between source and target distributions improving the object detector performance on the target domains when training and tests data belong to different distributions.<br>
If you want to use this code with your dataset, please follow the following guide. <br>
Please leave a star ⭐ and cite the following [paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314222000911?CMX_ID=&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED&utm_acid=170381815&utm_campaign=STMJ_AUTH_SERV_PUBLISHED&utm_in=DM270343&utm_medium=email&utm_source=AC_) if you use this repository for your project.
```
@article{PASQUALINO2022103487,
    title = {A multi camera unsupervised domain adaptation pipeline for object detection in cultural sites through adversarial learning and self-training},
    journal = {Computer Vision and Image Understanding},
    pages = {103487},
    year = {2022},
    issn = {1077-3142},
    doi = {https://doi.org/10.1016/j.cviu.2022.103487},
    url = {https://www.sciencedirect.com/science/article/pii/S1077314222000911},
    author = {Giovanni Pasqualino and Antonino Furnari and Giovanni Maria Farinella}
}
```

## Architecture

### Step 1
In this step the models is trained using synthetic labeled images (source domain) and unlabeled real images Hololens and GoPro (target domains). At the end of this step, the model is used to produce pseudo label on both Hololens and GoPro images.  
<img src='STMDA-RetinaNet-step1.png' width=100%/>

### Step 2
In this step the model is trained using synthetic labeled images (source domain) and pseudo label real images Hololens and GoPro (target domains) produced in the previous step. This is an iterative step and at the end of each step we produce better pseudo label on the target domains.
<img src='STMDA-RetinaNet-step2.png' width=100%/>

## Installation
You can use this repo following one of these three methods:<br>
NB: Detectron2 0.2.1 is required, installing other versions this code will not work.


### Google Colab
Quickstart here 👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/GiovanniPasq/506e9a535893de5ec287970af73968db/stmda-retinanet.ipynb)<br>
Or load and run the ```STMDA-RetinaNet.ipynb``` on Google Colab following the instructions inside the notebook.

### Detectron 2 on your PC
Follow the official guide to install [Detectron2 0.2.1](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)<br>
Or<br>
Download the official Detectron 0.2.1 from [here](https://github.com/facebookresearch/detectron2/archive/refs/tags/v0.2.1.zip)<br>
Unzip the file and rename it in detectron2<br>
run ```python -m pip install -e detectron2```

### Detectron2 via Dockerfile
Follow these instructions:
```
cd docker/
# Build 
docker build -t detectron2:v0 .

# Launch
docker run --gpus all -it --shm-size=8gb -v /home/yourpath/:/home/yourpath --name=name_container detectron2:v0
```
If you exit from the container you can restart it using:
```
docker start name_container
docker exec -it name_container /bin/bash
```

## Dataset
Dataset is available [here](https://iplab.dmi.unict.it/OBJ-MDA/OBJ_MDA.zip)

### Data Preparation
If you want to use this code with your dataset arrange the dataset in the format of COCO. Inside the script ```stmda_train.py``` register your dataset using: <br> ```register_coco_instances("dataset_name_source_training",{},"path_annotations","path_images")```<br>
```register_coco_instances("dataset_name_init_target_training",{},"path_annotations","path_images")```<br>
```register_coco_instances("dataset_name_init_target2_training",{},"path_annotations","path_images")```<br><br>

these are the paths where will be saved the annotations produced at the end of the step 1<br>
```register_coco_instances("dataset_name_target_training",{},"path_annotations","path_images")```<br> 
```register_coco_instances("dataset_name_target2_training",{},"path_annotations","path_images")```<br><br>

```register_coco_instances("dataset_name_target_test",{},"path_annotations","path_images")```<br>
```register_coco_instances("dataset_name_target_test2",{},"path_annotations","path_images")```<br>


### Training
Replace at the following path ```detectron2/modeling/meta_arch/``` the ```retinanet.py``` script with our ```retinanet.py```. <br>
Do the same for the ```fpn.py``` file at the path ```detectron2/modeling/backbone/```, ```evaltuator.py``` and ```coco_evaluation.py``` at ```detectron2/evaluation/```<br>
Inside the script ```stmda_train.py``` you can set the parameters for the second step training like number of iteration and threshold.<br>
Run the script ```stmda_train.py``` <br>
Trained models are available at these links: <br>
[MDA-RetinaNet](https://iplab.dmi.unict.it/OBJ-MDA/MDA-RetinaNet.pth)<br>
[STMDA-RetinaNet](https://iplab.dmi.unict.it/OBJ-MDA/STMDA-RetinaNet.pth) <br>
<br>
[MDA-RetinaNet-CycleGAN](https://iplab.dmi.unict.it/OBJ-MDA/MDA-RetinaNet-CycleGAN.pth)<br>
[STMDA-RetinaNet-CycleGAN](https://iplab.dmi.unict.it/OBJ-MDA/STMDA-RetinaNet-CycleGAN.pth)

### Testing
If you want to test the model load the new weights, set to 0 the number of iterations and run ```stmda_train.py```

### Results
<p>
  Results of baseline and feature alignment methods. S refers to Synthetic, H refers to Hololens and G to GoPro.
</p>

<table>
  <tr>
    <th>Model</th>
    <th>Source</th>
    <th>Target</th>
    <th>Test H</th>
    <th>Test G</th>
  </tr>
  <tr>
    <td>Faster RCNN</td>
    <td>S</td>
    <td>-</td>
    <td>7.61%</td>
    <td>30.39%</td>
  </tr>
  <tr>
    <td>RetinaNet</td>
    <td>S</td>
    <td>-</td>
    <td>14.10%</td>
    <td>37.13%</td>
  </tr>
  <tr>
    <td><a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf">DA-Faster RCNN</a></td>
    <td>S</td>
    <td>H+G merged</td>
    <td>10.53%</td>
    <td>48.23%</td>
  </tr>
  <tr>
    <td><a href= "https://openaccess.thecvf.com/content_CVPR_2019/papers/Saito_Strong-Weak_Distribution_Alignment_for_Adaptive_Object_Detection_CVPR_2019_paper.pdf"> StrongWeak</a></td>
    <td>S</td>
    <td>H+G merged</td>
    <td>26.68%</td>
    <td>48.55%</td>
  </tr>
  <td><a href= "https://arxiv.org/pdf/1911.07158.pdf">CDSSL</a></td>
    <td>S</td>
    <td>H+G merged</td>
    <td>28.66%</td>
    <td>45.33%</td>
  </tr>
  <tr>
    <td><a href="https://www.sciencedirect.com/science/article/pii/S0262885621000032?dgcid=rss_sd_all">DA-RetinaNet</a></td>
    <td>S</td>
    <td>H+G merged</td>
    <td>31.63%</td>
    <td>48.37%</td>
  </tr>
  <tr>
    <td>MDA-RetinaNet</td>
    <td>S</td>
    <td>H, G</td>
    <td>34.97%</td>
    <td>50.81%</td>
  </tr>
  <tr>
    <td>STMDA-RetinaNet</td>
    <td>S</td>
    <td>H, G</td>
    <td>54.36%</td>
    <td>59.51%</td>
  </tr>
</table>
<p>
Results of baseline and feature alignment methods combined with CycleGAN. H refers to Hololens while G to GoPro. "{G, H}" refers to synthetic images translated to the merged Hololens and GoPro domains.
</p>
<table>
  <tr>
    <th>Model</th>
    <th>Source</th>
    <th>Target</th>
    <th>Test H</th>
    <th>Test G</th>
  </tr>
  <tr>
    <td>Faster RCNN</td>
    <td>{G, H}</td>
    <td>-</td>
    <td>15.34%</td>
    <td>63.60%</td>
  </tr>
  <tr>
    <td>RetinaNet</td>
    <td>{G, H}</td>
    <td>-</td>
    <td>31.43%</td>
    <td>69.59%</td>
  </tr>
  <tr>
    <td><a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf">DA-Faster RCNN</a></td>
    <td>{G, H}</td>
    <td>H+G merged</td>
    <td>32.13%</td>
    <td>65.19%</td>
  </tr>
  <tr>
    <td><a href= "https://openaccess.thecvf.com/content_CVPR_2019/papers/Saito_Strong-Weak_Distribution_Alignment_for_Adaptive_Object_Detection_CVPR_2019_paper.pdf"> StrongWeak</a></td>
    <td>{G, H}</td>
    <td>H+G merged</td>
    <td>41.11%</td>
    <td>66.45%</td>
  </tr>
  <tr>
    <td><a href="https://www.sciencedirect.com/science/article/pii/S0262885621000032?dgcid=rss_sd_all">DA-RetinaNet</a></td>
    <td>{G, H}</td>
    <td>H+G merged</td>
    <td>52.07%</td>
    <td>71.14%</td>
  </tr>
  <td><a href= "https://arxiv.org/pdf/1911.07158.pdf">CDSSL</a></td>
    <td>{G, H}</td>
    <td>H+G merged</td>
    <td>53.06%</td>
    <td>71.17%</td>
  </tr>
  <tr>
    <td>MDA-RetinaNet</td>
    <td>{G, H}</td>
    <td>H, G</td>
    <td>58.11%</td>
    <td>71.39%</td>
  </tr>
  <tr>
    <td>STMDA-RetinaNet</td>
    <td>{G, H}</td>
    <td>H, G</td>
    <td>66.64%</td>
    <td>72.22%</td>
  </tr>
  
</table>

<!--```
@inproceedings{pasqualino2021multitarget,
  year = { 2021 },
  booktitle = { International Conference on Image Processing (ICIP) },
  title = { Unsupervised Multi-Target Domain Adaptation for Object Detection },
  author = { Giovanni Pasqualino, Antonino Furnari, Giovanni Maria Farinella },
}
```-->
## Other Works
[DA-RetinaNet](https://github.com/fpv-iplab/DA-RetinaNet) <br>
[Detectron2 implementation of DA-Faster RCNN](https://github.com/GiovanniPasq/DA-Faster-RCNN)
