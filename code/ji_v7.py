import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import sys
import json
sys.path.insert(1, '/project/train/src_repo/yolov7/')
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# ####### 参数设置
conf_thres = 0.3
iou_thres = 0.2
imgsz = 640
weights = "/project/train/models/exp/weights/last.pt"
device = '0'
stride = 32
names=["person", "short_sleeve_red", "short_sleeve_black", "short_sleeve_white", "short_sleeve_grey",
            "short_sleeve_green", "short_sleeve_blue", "short_sleeve_dark_blue", "long_sleeve_red",
            "long_sleeve_black", "long_sleeve_white", "long_sleeve_grey", "non_uniform",
            "other_uniform", "chef_hat_red", "chef_hat_black", "chef_hat_white", "peaked_cap_red",
            "peaked_cap_black", "peaked_cap_white", "peaked_cap_blue", "peaked_cap_beige",
            "disposable_cap_white", "disposable_cap_blue", "head", "other_hat", "apron_red",
            "apron_black", "apron_white", "apron_grey", "other_apron"]
def init():
    # Initialize
    global imgsz, device, stride
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model.half()  # to FP16
    model.eval()
    # model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    return model

def process_image(model, input_image=None, args=None, **kwargs):
    # Padded resize
    t0 = time.time()
    img0 = input_image
    img = letterbox(img0, new_shape=imgsz, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half()
#     img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]
    t1 = time.time()
    print(t1-t0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
    t2 = time.time()
    print(t2-t1)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    fake_result = {}

    fake_result["algorithm_data"] = {
       "is_alert": False,
       "target_count": 0,
       "target_info": []
   }
    fake_result["model_data"] = {"objects": []}
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                if int(cls)!=0:
                    fake_result["model_data"]['objects'].append({
                        "x":int(xyxy[0]),
                        "y":int(xyxy[1]),
                        "width":int(xyxy[2]-xyxy[0]),
                        "height":int(xyxy[3]-xyxy[1]),
                        "confidence":float(conf),
                        "name":names[int(cls)]
                    })
    fake_result ["algorithm_data"]["target_info"]=[]
    return json.dumps(fake_result, indent = 4)

if __name__ == '__main__':
    from glob import glob
    # Test API
    image_names = glob('/home/data/812/*.jpg')
    predictor = init()
    s = 0
    for image_name in image_names:
        print(image_name)
        img = cv2.imread(image_name)
        t1 = time.time()
        res = process_image(predictor, img)
        print(res)
        t2 = time.time()
        s += t2 - t1
        break
    print(1/(s/100))