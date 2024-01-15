import numpy as np
import torch

from yolov3.models.yolov3 import YOLOv3, YOLOv3Tiny
from yolov3.models.yolov3_edge import YOLOv3 as YOLOv3Edge
from yolov3.models.yolov3_cloud import YOLOv3 as YOLOv3Cloud


def conv_initializer(param):
    out_ch, in_ch, h, w = param.shape
    fan_in = h * w * in_ch
    scale = np.sqrt(2 / fan_in)

    w = scale * torch.randn_like(param)

    # n = scale * np.random.normal(size=param.numel())
    # w = torch.from_numpy(n).view_as(param)

    return w


def parse_conv_block(module, weights, offset, initialize):
    
    conv, bn, leakey = module

    params = [
        bn.bias,
        bn.weight,
        bn.running_mean,
        bn.running_var,
        conv.weight,
    ]

    for param in params:
        if initialize:
            if param is bn.weight:
                w = torch.ones_like(param)
            elif param is conv.weight:
                w = conv_initializer(param)
            else:
                w = torch.zeros_like(param)
        else:
            param_len = param.numel()
            w = torch.from_numpy(weights[offset : offset + param_len]).view_as(param)
            offset += param_len

        param.data.copy_(w)

    return offset


def parse_yolo_block(module, weights, offset, initialize):
    
    conv = module._modules["conv"]

    for param in [conv.bias, conv.weight]:
        if initialize:
            if param is conv.bias:
                w = torch.zeros_like(param)
            else:
                w = conv_initializer(param)
        else:
            param_len = param.numel()
            w = torch.from_numpy(weights[offset : offset + param_len]).view_as(param)
            offset += param_len

        param.data.copy_(w)

    return offset


def parse_yolo_weights(model, weights_path):
  
    with open(weights_path, "rb") as f:
        # skip header
        f.read(20)

        # read weights
        weights = np.fromfile(f, dtype=np.float32)

    offset = 0
    initialize = False


    for module in model.module_list:
        if module._get_name() == "Sequential":
            # conv / batch norm / leaky LeRU
            offset = parse_conv_block(module, weights, offset, initialize)

        elif module._get_name() == "resblock":
            # residual block
            for resblocks in module._modules["module_list"]:
                for resblock in resblocks:
                    offset = parse_conv_block(resblock, weights, offset, initialize)

        elif module._get_name() == "YOLOLayer":
            # YOLO Layer (one conv with bias) Initialization
            offset = parse_yolo_block(module, weights, offset, initialize)

        # the end of the weights file. turn the flag on
        initialize = offset >= len(weights)

def parse_yolo_weights_edge(model, weights_path):
  
    with open(weights_path, "rb") as f:
        # skip header
        f.read(20)

        weights = np.fromfile(f, dtype=np.float32)

    offset = 0
    initialize = False
    print(model)


    for module in model.module_list_share:
        if module._get_name() == "Sequential":
            # conv / batch norm / leaky LeRU
            offset = parse_conv_block(module, weights, offset, initialize)

        elif module._get_name() == "resblock":
            # residual block
            for resblocks in module._modules["module_list"]:
                for resblock in resblocks:
                    offset = parse_conv_block(resblock, weights, offset, initialize)

        elif module._get_name() == "YOLOLayer":
            # YOLO Layer (one conv with bias) Initialization
            offset = parse_yolo_block(module, weights, offset, initialize)
        
    for module in model.module_list_edgetail:
        if module._get_name() == "Sequential":
            # conv / batch norm / leaky LeRU
            offset = parse_conv_block(module, weights, offset, initialize)

        elif module._get_name() == "resblock":
            # residual block
            for resblocks in module._modules["module_list"]:
                for resblock in resblocks:
                    offset = parse_conv_block(resblock, weights, offset, initialize)

        elif module._get_name() == "YOLOLayer":
            # YOLO Layer (one conv with bias) Initialization
            offset = parse_yolo_block(module, weights, offset, initialize)



        initialize = offset >= len(weights)



def create_model(config):
    if config["model"]["name"] == "yolov3":
        model = YOLOv3(config["model"])
    elif config["model"]["name"] == "yolov3-edge":
        model = YOLOv3Edge(config["model"])
    elif config["model"]["name"] == "yolov3-cloud":
        model = YOLOv3Cloud(config["model"])
    else:
        model = YOLOv3Tiny(config["model"])

    return model
