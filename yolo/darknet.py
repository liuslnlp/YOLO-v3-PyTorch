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
    def __init__(self, cfg_filename, height=None):
        super(Darknet, self).__init__()
        self.blocks = load_cfg(cfg_filename)
        if height is not None:
            self.blocks[0]['height'] = height
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def load_darknet_weights(self, filename):
        self.header, self.seen = load_darknet_model(
            filename, self.blocks, self.module_list)

    def forward(self, x):
        detections = []
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        for i in range(len(modules)):
            module_type = (modules[i]["type"])
            if module_type in ("convolutional", "maxpool"):
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "upsample":
                scale_factor = self.module_list[i][0].buf
                x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
                outputs[i] = x

            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [n if n < 0 else n - i for n in map(int, layers)]
                maps = [outputs[i + n] for n in layers]
                x = torch.cat(maps, 1)
                outputs[i] = x

            elif module_type == "shortcut":
                frm = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+frm]
                outputs[i] = x

            elif module_type == 'yolo':
                x = self.module_list[i](x)
                detections.append(x)
                outputs[i] = outputs[i-1]
        return torch.cat(detections, 1)
