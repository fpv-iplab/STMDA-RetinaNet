import json
import glob
import csv
import os
import math
import pandas as pd

#Create json annotation file from output
def fix_annotations(start_json, prediction, output):

    s_json = start_json
    with open(s_json) as file:
        js_init = json.load(file)

    p_json = prediction
    with open(p_json) as file2:
        js_pred = json.load(file2)

    k = 0
    image_id = []
    annotations = []
    #formatting prediction
    for i in js_pred:
        i["segmentation"] =''
        i["'iscrowd'"] = 0
        i["bbox"][0] = math.ceil(i["bbox"][0])
        i["bbox"][1] = math.ceil(i["bbox"][1])
        i["bbox"][2] = math.ceil(i["bbox"][2])
        i["bbox"][3] = math.ceil(i["bbox"][3])
        i["area"] = i["bbox"][2] * i["bbox"][3]
        i["id"] = k
        del i["score"]
        annotations.append(i)
        image_id.append(i["image_id"])
        k+=1
    #selecting images with predictions
    '''
    images = []
    for i in js_init["images"]:
        if i["id"] in image_id:
            images.append(i)
    '''



    T_js = {"annotations":annotations, "categories":js_init["categories"], "images":js_init["images"]} #images
    with open(output, 'w') as outfile:
        json.dump(T_js, outfile)
