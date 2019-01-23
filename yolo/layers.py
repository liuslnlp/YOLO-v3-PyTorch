import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import predict_transform


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    """Placeholder"""

    def __init__(self, buf=None):
        super().__init__()
        self.buf = buf


class DetectionLayer(nn.Module):
    def __init__(self, inp_dim, anchors, num_classes):
        super().__init__()
        self.inp_dim = inp_dim
        self.anchors = anchors
        self.num_classes = num_classes
        #print("DetLay: ",inp_dim, anchors, num_classes)

    def forward(self, x):
        x = x.data
        prediction = predict_transform(
            x, self.inp_dim, self.anchors, self.num_classes)
        return prediction
