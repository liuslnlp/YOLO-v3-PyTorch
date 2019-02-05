import time
import torch 
import torch.nn as nn
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
from yolo.darknet import Darknet
from yolo.io import load_classes, load_colors
from yolo.util import plot_rectangle, sift_results
from yolo.preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
import platform

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Camera Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "PyTorch format weights file",
                        default = "yolov3.pkl", type = str)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    classes = load_classes('data/coco.names')
    colors = load_colors('data/pallete')
    
    
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(args.cfgfile, height=args.reso)
    model.load_state_dict(torch.load(args.weightsfile))
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    videofile = 'video.avi'
    
    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            output = model(img)
            output = sift_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
        
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            
#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            
            list(map(lambda x: plot_rectangle(x, orig_im, classes, colors), output))
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            
        else:
            break
    

    
    

