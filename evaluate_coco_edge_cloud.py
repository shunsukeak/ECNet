import argparse
from pathlib import Path

import torch
from yolov3.utils import utils as utils
# from yolov3.utils.coco_evaluator import COCOEvaluator2
from yolov3.utils.coco_evaluator import *
# from yolov3.utils.model import create_model, parse_yolo_weights
# from yolov3.utils.model_edge import create_model, parse_yolo_weights
from yolov3.utils.model import create_model, parse_yolo_weights, parse_yolo_weights_edge


#import time
#torch.cuda.synchronize()
#start = time.time()

#with torch.no_grad():
#  out = model_gpu(input_batch_gpu) maybe no need


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=Path, required=True, help="directory path to coco dataset"
    )
    parser.add_argument("--anno_path", type=Path, required=True, help="json filename")
    parser.add_argument(
        "--weights",
        type=Path,
        # default="weights/yolov3.weights",
        default="./train_output_test/ckpt/yolov3-edge_000300.ckpt",
        # default="weights/yolov3-tiny.weights",
        help="path to weights file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        # default="config/yolov3_coco.yaml",
        # default="config/yolov3tiny_coco.yaml",
        default="config/yolov3edge_coco.yaml",
        help="path to config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    config_edge = utils.load_config("config/yolov3edge_coco.yaml")
    config_cloud = utils.load_config("config/yolov3_coco.yaml")
    img_size = config_edge["test"]["img_size"]
    batch_size = 1

    
    device = utils.get_device(gpu_id=args.gpu_id)

    
    model_edge = create_model(config_edge)
    # parse_yolo_weights(model_edge, Path("weights/yolov3-tiny.weights"))
    # parse_yolo_weights_edge(model_edge, Path("./train_output_test/ckpt/yolov3-edge_000300.ckpt"))
    state = torch.load(args.weights)

    t = 0
    for module in model_edge.module_list_dummy:
        if t >= 9:
            model_edge.module_list_edgetail.append(model_edge.module_list_dummy[t])
        t+=1 


    model_edge.load_state_dict(state["model"], strict=True)
    print(f"Checkpoint file {args.weights} loaded.")

    model_edge = model_edge.to(device).eval()
    from torchsummary import summary 
    summary(model_edge,(3,416,416)) # summary(model,(channels,H,W))
    from mmcv.cnn.utils import flops_counter
    flops_counter.get_model_complexity_info(model_edge, (3, 416, 416))



    
    model_cloud = create_model(config_cloud)
    parse_yolo_weights(model_cloud, Path("weights/yolov3.weights"))
    model_cloud = model_cloud.to(device).eval()
    summary(model_cloud,(3,416,416)) # summary(model,(channels,H,W))
    from mmcv.cnn.utils import flops_counter
    flops_counter.get_model_complexity_info(model_cloud, (3, 416, 416))    


    
    from model import Featurecomp_FactorizedPrior
    comp = Featurecomp_FactorizedPrior()
    state = torch.load("/home/shunsukeakamatsu/Documents/pytorch_yolov3/output/feature_comp/lambda0.05/49_checkpoint.pth.tar")
    comp.load_state_dict(state["state_dict"])
    comp = comp.to(device)
    from torchsummary import summary 
    summary(comp,(256,416,416)) # summary(model,(channels,H,W))
    flops_counter.get_model_complexity_info(comp, (256, 416, 416)) 

 

    """Current edge cloud"""
    evaluator = COCOEvaluator_edgecloud(
        args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    )
    ap50_95, ap50 = evaluator.evaluate(model_edge, model_cloud,comp=comp)
    print(ap50_95, ap50)

    

if __name__ == "__main__":
    main()

#torch.cuda.synchronize()
#elapsed_time = time.time() - start
#print(elapsed_time, 'sec.')
