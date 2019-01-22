import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from .helper import create_modules
from .util import predict_transform
from .io import load_cfg, load_darknet_model


class Darknet(nn.Module):
    def __init__(self, cfg_filename):
        super(Darknet, self).__init__()
        self.blocks = load_cfg(cfg_filename)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def load_darknet_weights(self, filename):
        self.header, self.seen = load_darknet_model(filename, self.blocks, self.module_list)

    def forward(self, x):
        detections = []
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i in range(len(modules)):

            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "maxpool":

                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "upsample":
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                outputs[i] = x

            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x

            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int(self.net_info["height"])

                #Get the number of classes
                num_classes = int(modules[i]["classes"])

                #Output the result
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes)

                if type(x) == int:
                    continue

                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]
        return detections

