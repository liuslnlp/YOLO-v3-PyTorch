import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
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
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "PyTorch format weights file",
                        default = "yolov3.pkl", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    parser.add_argument("--cuda", dest = "use_cuda", help = "Whether use CUDA. `y` or `n`",
                        default = 'y', type = str, )
    
    return parser.parse_args()

def main():
    args = arg_parse()
    scales = args.scales
    weights_file = args.weightsfile
    images = args.images
    use_cuda = args.use_cuda
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()
    if use_cuda == 'n':
        CUDA = False

    print("Loading network.....")
    model = Darknet(args.cfgfile, height=args.reso)
    # model.load_weights(args.weightsfile)
    model.load_state_dict(torch.load(weights_file))
    print("Network successfully loaded")
    #model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()


    model.eval()

    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(
            img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()

    batches = list(
        map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    classes = load_classes('data/coco.names')
    
    num_classes = 80

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)*batch_size,
                                                              len(im_batches))])) for i in range(num_batches)]
    i = 0


    start_det_loop = time.time()

    objs = {}
    output = []
    for batch in im_batches:
        #load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        # print(batch.shape) torch.Size([1, 3, 416, 416])

        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(batch)

#        prediction = prediction[:,scale_indices]

        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence
        #clubbing these ops in one loop instead of two.
        #loops are slower than vectorised operations.

        prediction = sift_results(
            prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        end = time.time()
        prediction[:, 0] += i*batch_size

        output.append(prediction)

        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])]
                    for x in output[-1] if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        if CUDA:
            torch.cuda.synchronize()

    if len(output) == 0:
        raise NameError("No detections were made")
    output = torch.cat(output)

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 1].view(-1, 1))/2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(
            output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(
            output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()

    class_load = time.time()
    colors = load_colors('data/pallete')
    draw = time.time()
    for out in output:
        idx = int(out[0])
        plot_rectangle(out, orig_ims[idx], classes, colors)

    if platform.system() == 'Windows':
        det_names = pd.Series(imlist).apply(
            lambda x: "{}\\det_{}".format(args.det, x.split("\\")[-1]))
    else:
        det_names = pd.Series(imlist).apply(
            lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

    list(map(cv2.imwrite, det_names, orig_ims))

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format(
        "Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format(
        "Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format(
        "Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format(
        "Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    # torch.save(model.state_dict(), 'yolov3.pkl')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()