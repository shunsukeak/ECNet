import json
import tempfile

import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
from yolov3.datasets.coco import COCODataset
from yolov3.utils.utils import postprocess
import math


class COCOEvaluator:
    def __init__(self, dataset_dir, anno_path, img_size, batch_size):
        self.conf_threshold = 0.005

        self.nms_threshold = 0.45
        # Dataset を作成する。
        self.dataset = COCODataset(dataset_dir, anno_path, img_size=img_size, augmentation = None,)
        # DataLoader を作成する。
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size
        )

    def evaluate_results(self, coco, detections):
        tf = tempfile.NamedTemporaryFile(mode="w")
        tf.write(json.dumps(detections))
        img_ids = [x["image_id"] for x in detections]

        cocoDt = coco.loadRes(tf.name)
        cocoEval = COCOeval(coco, cocoDt, "bbox")
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]

        return ap50_95, ap50

   
    
    def output_to_dict(self, output, img_id):
        detection = []
        for x1, y1, x2, y2, obj_conf, class_conf, label in output:

            bbox = {
                    "image_id": int(img_id),
                    "category_id": self.dataset.category_ids[int(label)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(obj_conf * class_conf),
                }

            detection.append(bbox)

          
        return detection


    def evaluate(self, model, comp=None):
        model.eval()
        device = next(model.parameters()).device

        if comp is not None:
            comp.eval()

        detections = []
        for inputs, labels, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]
            # with torch.amp.autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                if comp is not None:
                    mid = model(inputs, test_enc=True)
                    out_net = comp(mid)
                    outputs = model(out_net["x_hat"], test_dec=True)
                    # outputs = model(mid, test_dec=True)
                    outputs = postprocess(
                        outputs, self.conf_threshold, self.nms_threshold, pad_infos
                    )

                else:
                    outputs = model(inputs)
                    outputs = postprocess(
                        outputs, self.conf_threshold, self.nms_threshold, pad_infos
                    )

            for output, img_id in zip(outputs, img_ids):
                detections += self.output_to_dict(output, img_id)

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        return ap50_95, ap50

class COCOEvaluator_edgecloud:
    def __init__(self, dataset_dir, anno_path, img_size, batch_size):
        self.conf_threshold = 0.005

        self.nms_threshold = 0.45
        self.dataset = COCODataset(dataset_dir, anno_path, img_size=img_size, augmentation = None,)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size
        )

    def evaluate_results(self, coco, detections):
        tf = tempfile.NamedTemporaryFile(mode="w")
        tf.write(json.dumps(detections))
        img_ids = [x["image_id"] for x in detections]

        cocoDt = coco.loadRes(tf.name)
        cocoEval = COCOeval(coco, cocoDt, "bbox")
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]

        return ap50_95, ap50
    
    def output_to_dict(self, output, img_id):
        detection = []
        for x1, y1, x2, y2, obj_conf, class_conf, label in output:

            bbox = {
                    "image_id": int(img_id),
                    "category_id": self.dataset.category_ids[int(label)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(obj_conf * class_conf),
                }

            detection.append(bbox)

           
        return detection


    def evaluate(self, model_edge, model_cloud, comp=None):
        model_edge.eval()
        model_cloud.eval()
        device = next(model_edge.parameters()).device

        if comp is not None: 
            comp.eval()




        detections = []
        bpp = 0
        cloud = 0
        for inputs, labels, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]
            # with torch.amp.autocast('cuda', dtype=torch.float16):
            with torch.no_grad():
                outputs_edge, feature_edge_head = model_edge(inputs, usecloud=True)

                # confidence = outputs_edge[:, :, 4:5] * outputs_edge[:, :, 5:]
                obj = outputs_edge[:, :, 4:5]
                cls = outputs_edge[:, :, 5:]

                # ind = obj>0.2
                ind = obj>=0
                ind = ind[:,:,0]
                confidence = obj[ind]*cls[ind]
                # print(confidence.max(),confidence.min(),confidence.mean())
                try:
                    confidence.max()
                    print(confidence.max())
                    fg = True
                except:
                    fg = False
                
                # if comp is not None:
                if fg and confidence.max() < 1.5:
                    mid = model_cloud(inputs, test_enc=True)
                    out_net = comp(mid)

                    N, _, H, W = inputs.size()
                    num_pixels = N * H * W
                    bpp += sum(
                        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                        for likelihoods in out_net["likelihoods"].values()
                    )
                    outputs = model_cloud(out_net["x_hat"], test_dec=True)
                    outputs = postprocess(
                        outputs, self.conf_threshold, self.nms_threshold, pad_infos
                    )

                    cloud+=1

                else:
                    outputs = postprocess(
                        outputs_edge, self.conf_threshold, self.nms_threshold, pad_infos
                    )

            for output, img_id in zip(outputs, img_ids):
                detections += self.output_to_dict(output, img_id)

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        print(bpp/5000)
        print(cloud, "/5000 : Cloud")
        print(9.481 + (cloud/5000)* 33.048)

        return ap50_95, ap50
    
    def compression(input, fs, qa=30):
        from PIL import Image
        from io import BytesIO
        import os
        import torch
        import torchvision
        import torchvision.transforms as T
        from PIL import Image
        transform = T.ToPILImage()
        COMPRESS_QUALITY = qa 

        jpeg_imgefile = 'jpeg_image.jpg'

        
        input = transform(torch.reshape(input,[3,416,416]))
        file_name = os.path.splitext(os.path.basename(jpeg_imgefile))[0]
        
        im = input

        im_io = BytesIO()
        im.save(im_io, 'JPEG', quality = COMPRESS_QUALITY)
        with open('comp_' + file_name + '.jpg', mode='wb') as outputfile:
            outputfile.write(im_io.getvalue())
            fs += os.path.getsize('comp_' + file_name + '.jpg')

        im = Image.open('comp_' + file_name + '.jpg')

        input = torchvision.transforms.functional.to_tensor(im)
        
        from torchvision.utils import save_image
        save_image(input, "./masked_image_comp.jpg")

        return input, fs




def encode_bboxes(bboxes, pad_info):
    scale_x, scale_y, dx, dy = pad_info

    bboxes *= np.array([scale_x.item(), scale_y.item(), scale_x.item(), scale_y.item()])
    bboxes[0] += dx.item()
    bboxes[1] += dy.item()

    return bboxes
