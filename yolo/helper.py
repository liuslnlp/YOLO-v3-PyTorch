import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .layers import EmptyLayer, MaxPoolStride1, DetectionLayer


def create_modules(blocks):
    """A help function to build module list by blocks.
    Parameters
    ----------
    blocks : A list to describe hyperparameters in different layers.
    blocks[i] is a dict, blocks[i][hyperparameter_key] = hyperparameter_val

    Return
    ----------
    (net_info, module_list), net_info is a dict object, includes some information
    like `batch`, `input width` and so on.
    module_list is a stack-like struct(nn.ModuleList), includes the net layers.
    """
    # Captures the information about the input and pre-processing
    net_info = blocks[0]

    module_list = nn.ModuleList()

    # indexing blocks helps with implementing route  layers (skip connections)
    index = 0

    prev_filters = 3

    output_filters = []

    for x in blocks:
        module = nn.Sequential()

        if (x["type"] == "net"):
            continue

        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation.
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        #If it's an upsampling layer
        #We use Bilinear2dUpsampling

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = EmptyLayer(stride)
            module.add_module("upsample_{}".format(index), upsample)

        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            #Start  of a route
            start = int(x["layers"][0])

            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            #Positive anotation
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index +
                                         start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(index), maxpool)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            inp_dim = int(net_info["height"])
            num_class = int(x["classes"])
            detection = DetectionLayer(inp_dim, anchors, num_class)
            module.add_module("Detection_{}".format(index), detection)

        else:
            print("Creat model fail!")
            assert False

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1

    return (net_info, module_list)
