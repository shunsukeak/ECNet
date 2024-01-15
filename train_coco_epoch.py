import argparse
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from yolov3.datasets.coco import COCODataset

from yolov3.utils import utils as utils
from yolov3.utils.model import create_model, parse_yolo_weights
from yolov3.utils.model_edge import create_model, parse_yolo_weights, parse_yolo_weights_edgetail 
from yolov3.models.yolov3_edge import resblock 

import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import gc
import os
from tqdm import tqdm
from yolov3.utils.coco_evaluator import *



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_dir", type=Path, default="./data/COCO",
        help="directory path to coco dataset",
    )
    
    parser.add_argument(
        "--anno_path", type=Path, default="./data/COCO/annotations_trainval2017/annotations/instances_train2017.json",
        help="json filename",
    )
   
    parser.add_argument(
        --weights", type=Path, default="weights/darknet53.conv.74
        help="path to darknet weights file (.weights) or checkpoint file (.pth)",
    )

    parser.add_argument(
        #"--config", type=Path, default="config/yolov3_coco.yaml",
        "--config", type=Path, default="./config/yolov3edge_coco.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id to use")
    parser.add_argument(
        "--save_dir", type=Path, default="train_output_test",
        help="directory where checkpoint files are saved",
    )
    # fmt: on
    args = parser.parse_args()

    return args


def remove_sequential(network):
    for layer in network.children():
        # print(type(layer))
        if type(layer) == resblock or type(layer) == torch.nn.Sequential or type(layer) ==  torch.nn.modules.container.ModuleList: # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer)
        if list(layer.children()) == []: # if leaf node, add it to list
            # print("leaf---", layer)
            if isinstance(layer, torch.nn.BatchNorm2d):
                if hasattr(layer, 'weight'):
                    layer.weight.requires_grad_(False)
                if hasattr(layer, 'bias'):
                    layer.bias.requires_grad_(False)

                layer.eval()


def build_optimizer(config, model):

    base_lr = config["train"]["lr"] 
    params = []

    
    for value in model.module_list_share.parameters(): 
        value.requires_grad = False

    for value in model.connect.parameters(): 
        params.append(value)

    for value in model.module_list_edgetail.parameters(): 
        params.append(value)
    

    optimizer = torch.optim.Adam(
        params,
        lr=base_lr
    )

    return optimizer


def build_scheduler(config, optimizer):
    
    factor = config["train"]["factor"]
    steps = config["train"]["steps"]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=factor)


    return scheduler


def repeater(dataloader):
    for loader in repeat(dataloader):
        for data in loader:
            yield data


def get_train_size():
    return np.random.randint(10, 20) * 32


def print_info(info, max_iter):
    print(f"iteration: {info['iter']}/{max_iter}", end=" ")
    print(f"image size: {info['img_size']}", end=" ")
    print(f"[Loss] total: {info['loss_total']:.2f}", end=" ")
    print(f"xy: {info['loss_xy']:.2f}", end=" ")
    print(f"wh: {info['loss_wh']:.2f}", end=" ")
    print(f"object: {info['loss_obj']:.2f}", end=" ")
    print(f"class: {info['loss_cls']:.2f}")


def vid_to_coco(labels):
    COCO_label = [1,2,3,4,5,6,14,16,17,18,20,21,22] 
    VID_label = [3,6,18,0,5,25,4,8,14,21,10,2,29] 
    # COCO_label = [n+1 for n in COCO_label]
    # VID_label = [n+1 for n in VID_label]

    for i in range(int(labels.shape[0])):
        class_ids = labels[i][:,0]
        bboxs = labels[i][:,1:]
        for j in range(len(class_ids)):
            if class_ids[j] == 0 and bboxs[j].sum() == 0:
                continue
            elif class_ids[j] in VID_label:
                ind = VID_label.index(class_ids[j])
                class_ids[j] = COCO_label[ind]
            else:
                class_ids[j] == 0
                bboxs[j] = torch.zeros_like(bboxs)
                
        labels[i][:,0] = class_ids
        labels[i][:,1:] = bboxs
    
    return labels


def train_val(model, dataloader, optimizer, scheduler, device, config, epoch, train=False):
    
    lossdict_total = {"total":0}
    if train:
        model.train()
        # remove_sequential(model)
    else:
        model.eval()
    for i, data in enumerate(tqdm(dataloader)):

        imgs, labels, pad_infos, _ = data
        imgs = imgs.to(device)

        """for vid"""
        # old_labels = labels.to(device)
        # labels = vid_to_coco(old_labels)

        labels = labels.to(device)

        # from torchvision.utils import save_image
        # save_image(imgs, "test.jpg")

        
        if train:
            loss = model(imgs, labels)
        else:
            with torch.no_grad():
                loss = model(imgs, labels)
                
       
        
        if train:
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        lossdict = model.loss_dict
        lossdict_total["total"] += loss.data

        for k in lossdict:
            if k in lossdict_total:
                lossdict_total[k] += lossdict[k]
            else:
                lossdict_total[k] = lossdict[k]

        if train:
            if (i+1) % 1000 == 0:
                lossdict_average = {}
                for k in lossdict_total:
                    lossdict_average[k] = float(lossdict_total[k]) / (i+1)
                print(lossdict_average)

        
        del imgs, labels, loss           
        torch.cuda.empty_cache()
        if (i+1) % 50 == 0:
            torch.cuda.empty_cache()
        if (i+1) % 250 == 0:
            torch.cuda.empty_cache() 
            gc.collect()  
    
    lossdict_average = {}
    for k in lossdict_total:
        lossdict_average[k] = float(lossdict_total[k]) / (i+1)
    print(lossdict_average)

    return lossdict_average


def main():
    args = parse_args()

    
    config = utils.load_config(args.config)
    img_size = config["train"]["img_size"]
    batch_size = config["train"]["batch_size"]
    max_epoch = config["train"]["epoch"]

    
    device = utils.get_device(gpu_id=args.gpu_id)
    print(device)

    
    state = torch.load(args.weights) if args.weights.suffix == ".ckpt" else None

    model = create_model(config)
    if state:
        # model.load_state_dict(state["model"])
        print(state["model"])
        model.load_state_dict(state["model"], strict=False)
        print(f"Checkpoint file {args.weights} loaded.")
    else:
        parse_yolo_weights(model, args.weights)
        parse_yolo_weights_edgetail(model, "/")
        print(f"Darknet weights file {args.weights} loaded.")

    t = 0
    for module in model.module_list_dummy:
        if t >= 9:
            model.module_list_edgetail.append(model.module_list_dummy[t])
        t+=1 

    model = model.to(device).train()

    
    optimizer = build_optimizer(config, model)

   
    scheduler = build_scheduler(config, optimizer)

    start_epoch = 1
    #start_epoch = 100
    history = []

    if state:
       
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        history = state["history"]

    train_dataset = COCODataset(
        args.dataset_dir,
        args.anno_path,
        img_size=img_size,
        augmentation=config["augmentation"],
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True,  pin_memory=True)

    val_dataset = COCODataset(
        args.dataset_dir,
        Path("./data/COCO/annotations_trainval2017/annotations/instances_val5k.json"),
        img_size=config["test"]["img_size"],
        augmentation=None
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["test"]["batch_size"], num_workers=2, shuffle=False,  pin_memory=True)

    evaluator = COCOEvaluator(
            args.dataset_dir, Path("./data/COCO/annotations_trainval2017/annotations/instances_val5k.json"), img_size=config["test"]["img_size"], batch_size=config["test"]["batch_size"]
        )
    

  
    os.makedirs(args.save_dir/"ckpt", exist_ok=True)
    os.makedirs(args.save_dir/"history", exist_ok=True)

    torch.backends.cudnn.benchmark = True

    for epoch in range(start_epoch, max_epoch + 1):

        """training-------------------------------------------------------------------------------"""
        lossdict_average = train_val(model, train_dataloader, optimizer, scheduler, device, config, epoch, train=True)
        """---------------------------------------------------------------------------------------"""

        """validation-----------------------------------------------------------------------------"""
        lossdict_average_val = train_val(model, val_dataloader, optimizer, scheduler, device, config, epoch, train=False)
        """----------------------------------------------------------------------------------------"""

        """test------------------------------------------------------------------------------------"""
        ap50_95, ap50 = evaluator.evaluate(model.eval())
        """----------------------------------------------------------------------------------------"""


        lossdict_average['epoch'] = epoch
        lossdict_average['accracy'] = ap50
        for k in lossdict_average_val:
            lossdict_average[k+'_val'] = lossdict_average_val[k]
        history.append(lossdict_average)

        model_name = config["model"]["name"]
       
        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
        }
        state_dict_path = args.save_dir /"ckpt"/ f"{model_name}_{epoch:06d}.ckpt"
        history_path = args.save_dir /"history"/ f"history_{epoch:06d}.csv"

        torch.save(state_dict, state_dict_path)

        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Training state saved. checkpoints: {state_dict_path}, loss history: {history_path}."
        )


        with open(str(args.save_dir)+'/'+str(epoch).zfill(5)+'.pkl', "wb") as tf:
            pickle.dump(history,tf)

        hor = [ item["epoch"] for item in history]

        ver_total_train = [item["total"] for item in history]
        ver_total_val = [item["total_val"] for item in history]
        val_ver_acc = [ item["accracy"] for item in history]

        fig = plt.figure()
        ax1 = fig.subplots()
        ax2 = ax1.twinx()

        ax1.plot(hor, ver_total_train, marker="o", color = "blue", linestyle = "--",label="train_loss")
        ax1.plot(hor, ver_total_val, marker="o", color = "red", linestyle = "--",label="val_loss")
        ax2.plot(hor, val_ver_acc, marker=".", color = "k", linestyle = "-",label="accuracy")
        ax1.set_ylabel('loss')
        ax2.set_ylabel("acc")
        ax1.set_xlabel('epoch')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')
        plt.savefig(str(args.save_dir)+'/history/'+str(epoch).zfill(5)+'.jpg')
        plt.clf()
        plt.close()



if __name__ == "__main__":
    main()
