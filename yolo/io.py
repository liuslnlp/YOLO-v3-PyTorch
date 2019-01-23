# Input and Output model.

import pickle as pkl
import torch
import numpy as np

def load_classes(namesfile):
    """Load classes file, which class names split by \n"""
    with open(namesfile, 'r') as f:
        names = f.read().split("\n")[:-1]
    return names

def load_colors(filename):
    """Load colors map."""
    with open(filename, "rb") as f:
        colors = pkl.load(f)
    return colors
    
def load_cfg(filename):
    """Load net structure config.
    Paramaters
    ----------
    filename : net structure config file(*.cfg).

    Return
    ----------
    List. lst[i] represent the ith layer's parameters.

    Examples
    ----------
    config file:
    [net]
    batch=1
    subdivisions=1
    [convolutional]
    ...

    blocks:
    blocks[0]['type'] = 'net'
    blocks[0]['batch'] = 1
    blocks[0]['subdivisions'] = 1
    blocks[1]['type'] = convolutional
    """
    blocks = []
    block = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip(' \n')
        # Get rid of blank line and comment. 
        if len(line)==0 or line[0] == '#':
            continue
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    if len(block) != 0:
        blocks.append(block)
    return blocks

def load_darknet_model(filename, blocks, module_list):
    """Load darknet style model weights file.
    Paramaters
    ----------
    filename : *.weights.
    blocks : module structure.
    module_list : nn.ModuleList
    """
    fp = open(filename, "rb")

    #The first 4 values are header information
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number
    # 4. IMages seen
    header = np.fromfile(fp, dtype=np.int32, count=5)
    header = torch.from_numpy(header)
    seen = header[3]
    #The rest of the values are the weights
    # Let's load them up
    weights = np.fromfile(fp, dtype=np.float32)
    ptr = 0
    for i in range(len(module_list)):
        module_type = blocks[i + 1]["type"]
        if module_type == "convolutional":
            model = module_list[i]
            try:
                batch_normalize = int(blocks[i+1]["batch_normalize"])
            except:
                batch_normalize = 0
            conv = model[0]
            if (batch_normalize):
                bn = model[1]
                #Get the number of weights of Batch Norm Layer
                num_bn_biases = bn.bias.numel()
                #Load the weights
                bn_biases = torch.from_numpy(
                    weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases
                bn_weights = torch.from_numpy(
                    weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases
                bn_running_mean = torch.from_numpy(
                    weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases
                bn_running_var = torch.from_numpy(
                    weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases
                #Cast the loaded weights into dims of model weights.
                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)
                #Copy the data to model
                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)
            else:
                #Number of biases
                num_biases = conv.bias.numel()
                #Load the weights
                conv_biases = torch.from_numpy(
                    weights[ptr: ptr + num_biases])
                ptr = ptr + num_biases
                #reshape the loaded weights according to the dims of the model weights
                conv_biases = conv_biases.view_as(conv.bias.data)
                #Finally copy the data
                conv.bias.data.copy_(conv_biases)
            #Let us load the weights for the Convolutional layers
            num_weights = conv.weight.numel()
            #Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
            ptr = ptr + num_weights
            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)
    return (header, seen)