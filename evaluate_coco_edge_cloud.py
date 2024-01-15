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

    # 設定ファイルを読み込む。
    # config_edge = utils.load_config("config/yolov3tiny_coco.yaml")
    config_edge = utils.load_config("config/yolov3edge_coco.yaml")
    config_cloud = utils.load_config("config/yolov3_coco.yaml")
    img_size = config_edge["test"]["img_size"]
    batch_size = 1

    # デバイスを作成する。
    device = utils.get_device(gpu_id=args.gpu_id)

    # モデルを作成する。
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



    # モデルを作成する。
    model_cloud = create_model(config_cloud)
    parse_yolo_weights(model_cloud, Path("weights/yolov3.weights"))
    model_cloud = model_cloud.to(device).eval()
    summary(model_cloud,(3,416,416)) # summary(model,(channels,H,W))
    from mmcv.cnn.utils import flops_counter
    flops_counter.get_model_complexity_info(model_cloud, (3, 416, 416))    


    # compmression model load
    from model import Featurecomp_FactorizedPrior
    comp = Featurecomp_FactorizedPrior()
    state = torch.load("/home/shunsukeakamatsu/Documents/pytorch_yolov3/output/feature_comp/lambda0.05/49_checkpoint.pth.tar")
    comp.load_state_dict(state["state_dict"])
    comp = comp.to(device)
    from torchsummary import summary 
    summary(comp,(256,416,416)) # summary(model,(channels,H,W))
    flops_counter.get_model_complexity_info(comp, (256, 416, 416)) 

    # evaluator = COCOEvaluator2(
    #     args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    # )
    # #  This is for edge-cloud
    # evaluator = COCOEvaluator4(
    #     args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    # )
    #  #This is for V3 only
    # evaluator = COCOEvaluator5(
    #     args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    # )
    # evaluator = COCOEvaluator6(
    #     args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    # )

    """Current edge cloud"""
    evaluator = COCOEvaluator_edgecloud(
        args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    )
    ap50_95, ap50 = evaluator.evaluate(model_edge, model_cloud,comp=comp)
    print(ap50_95, ap50)

    # x=0
    # thres1_list = [0.003]
    # thres1_list = [0.003, 0.0007, 0.005, 0]
    # thres1_list = [0.0003]
    # long = len(thres1_list)
    # for i in range(long):
    #     x = thres1_list[i]
    # ap50_95_1, ap50_1, fs_1 = evaluator.evaluate(model_edge, model_cloud, q=60, thres1=x)
    # ap50_95_2, ap50_2, fs_2 = evaluator.evaluate(model_edge, model_cloud, q=40, thres1=x)
    # ap50_95_3, ap50_3, fs_3 = evaluator.evaluate(model_edge, model_cloud, q=20, thres1=x)
    # ap50_95_4, ap50_4, fs_4 = evaluator.evaluate(model_edge, model_cloud, q=15, thres1=x)
    # ap50_95_5, ap50_5, fs_5 = evaluator.evaluate(model_edge, model_cloud, q=10, thres1=x)
    # ap50_95_6, ap50_6, fs_6 = evaluator.evaluate(model_edge, model_cloud, q=5, thres1=x)
        
        # ap50_95_1, ap50_1, fs_1 = evaluator.evaluate(model_edge, model_cloud, q=100, thres1=x)
        # ap50_95_2, ap50_2, fs_2 = evaluator.evaluate(model_edge, model_cloud, q=80, thres1=x)
        # ap50_95_3, ap50_3, fs_3 = evaluator.evaluate(model_edge, model_cloud, q=60, thres1=x)
        # ap50_95_4, ap50_4, fs_4 = evaluator.evaluate(model_edge, model_cloud, q=40, thres1=x)
        # ap50_95_5, ap50_5, fs_5 = evaluator.evaluate(model_edge, model_cloud, q=20, thres1=x)
        # ap50_95_6, ap50_6, fs_6 = evaluator.evaluate(model_edge, model_cloud, q=5, thres1=x)

        # print("thres=", x, ap50_95_1, ap50_1, fs_1)
        # print("thres=", x, ap50_95_2, ap50_2, fs_2)
        # print("thres=", x, ap50_95_3, ap50_3, fs_3)
        # print("thres=", x, ap50_95_4, ap50_4, fs_4)
        # print("thres=", x, ap50_95_5, ap50_5, fs_5)
        # print("thres=", x, ap50_95_6, ap50_6, fs_6)
        
    # ap50_95_1, ap50_1, fs_1 = evaluator.evaluate(model_edge, model_cloud, q=90, thres1=x)
    # ap50_95_2, ap50_2, fs_2 = evaluator.evaluate(model_edge, model_cloud, q=80, thres1=x)
    # ap50_95_3, ap50_3, fs_3 = evaluator.evaluate(model_edge, model_cloud, q=70, thres1=x)
    # ap50_95_4, ap50_4, fs_4 = evaluator.evaluate(model_edge, model_cloud, q=50, thres1=x)
    # ap50_95_5, ap50_5, fs_5 = evaluator.evaluate(model_edge, model_cloud, q=40, thres1=x)
    # ap50_95_6, ap50_6, fs_6 = evaluator.evaluate(model_edge, model_cloud, q=30, thres1=x)

    # print("thres=", x, ap50_95_1, ap50_1, fs_1)
    # print("thres=", x, ap50_95_2, ap50_2, fs_2)
    # print("thres=", x, ap50_95_3, ap50_3, fs_3)
    # print("thres=", x, ap50_95_4, ap50_4, fs_4)
    # print("thres=", x, ap50_95_5, ap50_5, fs_5)
    # print("thres=", x, ap50_95_6, ap50_6, fs_6)
    # bpp1 = (fs_1*8)/(416*416*5000)
    # bpp2 = (fs_2*8)/(416*416*5000)
    # bpp3 = (fs_3*8)/(416*416*5000)
    # bpp4 = (fs_4*8)/(416*416*5000)
    # bpp5 = (fs_5*8)/(416*416*5000)
    # bpp6 = (fs_6*8)/(416*416*5000)
    # print(ap50_95_1, ap50_1, fs_1, bpp1)
    # print(ap50_95_2, ap50_2, fs_2, bpp2)
    # print(ap50_95_3, ap50_3, fs_3, bpp3)
    # print(ap50_95_4, ap50_4, fs_4, bpp4)
    # print(ap50_95_5, ap50_5, fs_5, bpp5)
    # print(ap50_95_6, ap50_6, fs_6, bpp6)

    # #This is V3 only
    # ap50_95_1, ap50_1, fs_1 = evaluator.evaluate(model_edge, model_cloud, q=15)
    # ap50_95_2, ap50_2, fs_2 = evaluator.evaluate(model_edge, model_cloud, q=13)
    # ap50_95_3, ap50_3, fs_3 = evaluator.evaluate(model_edge, model_cloud, q=12)
    # ap50_95_4, ap50_4, fs_4 = evaluator.evaluate(model_edge, model_cloud, q=10)
    # ap50_95_5, ap50_5, fs_5 = evaluator.evaluate(model_edge, model_cloud, q=7)
    # ap50_95_6, ap50_6, fs_6 = evaluator.evaluate(model_edge, model_cloud, q=5)

    # bpp1 = (fs_1*8)/(416*416*5000)
    # bpp2 = (fs_2*8)/(416*416*5000)
    # bpp3 = (fs_3*8)/(416*416*5000)
    # bpp4 = (fs_4*8)/(416*416*5000)
    # bpp5 = (fs_5*8)/(416*416*5000)
    # bpp6 = (fs_6*8)/(416*416*5000)
    # print(ap50_95_1, ap50_1, fs_1, bpp1)
    # print(ap50_95_2, ap50_2, fs_2, bpp2)
    # print(ap50_95_3, ap50_3, fs_3, bpp3)
    # print(ap50_95_4, ap50_4, fs_4, bpp4)
    # print(ap50_95_5, ap50_5, fs_5, bpp5)
    # print(ap50_95_6, ap50_6, fs_6, bpp6)

if __name__ == "__main__":
    main()

#torch.cuda.synchronize()
#elapsed_time = time.time() - start
#print(elapsed_time, 'sec.')
