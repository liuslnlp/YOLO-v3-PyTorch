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
    def __init__(self):
        super().__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        prediction = predict_transform(
            x, inp_dim, self.anchors, num_classes)
        return prediction