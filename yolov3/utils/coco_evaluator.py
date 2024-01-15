import json
import tempfile

import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
# from yolov3.datasets.coco_vid import COCODataset
from yolov3.datasets.coco import COCODataset
from yolov3.utils.utils import postprocess
import math


class COCOEvaluator:
    def __init__(self, dataset_dir, anno_path, img_size, batch_size):
        # MS COCO の評価を行うときに使用する閾値
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

    def output_to_dict_vid(self, output, img_id):
        detection = []
        for x1, y1, x2, y2, obj_conf, class_conf, label in output:

            # label = int(label.item()) + 1
            # COCO_label = [2,3,4,5,6,7,15,17,18,19,21,22,23]
            # VID_label = [1,3,4,5,6,7,9,11,15,19,22,26,30]

            label = int(label.item())
            COCO_label = [1,2,3,4,5,6,14,16,17,18,20,21,22]
            # VID_label = [0,2,3,4,5,6,8,10,14,18,21,25,29]
            VID_label = [3,6,18,0,5,25,4,8,14,21,10,2,29]

            if label in COCO_label:
            # if(np.any(label == COCO_label)):
                ind = COCO_label.index(label)
                vid_label = VID_label[ind]

                # vid_label -= 1

                # print(label, vid_label, ind)

                bbox = {
                    "image_id": int(img_id),
                    "category_id": self.dataset.category_ids[int(vid_label)],
                    # "category_id": self.dataset.category_ids[int(COCO_label)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(obj_conf * class_conf),
                }

                detection.append(bbox)

        return detection
    
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

            # if (label == 0): #ラベルが0つまりpersonのとき、(MOT17用)
            #     bbox = {
            #         "image_id": int(img_id),
            #         "category_id": self.dataset.category_ids[int(label)],
            #         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            #         "score": float(obj_conf * class_conf),
            #     }

            #     detection.append(bbox)

            # else:
            #     print(label)
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
                        # 順伝搬を行う。
                    outputs = model(inputs)
                        # 後処理を行う。
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
        # MS COCO の評価を行うときに使用する閾値
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
                        # 後処理を行う。
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
        # コンフィグ
        COMPRESS_QUALITY = qa # 圧縮のクオリティ

        # JPEG形式とPNG形式の画像ファイルを用意
        jpeg_imgefile = 'jpeg_image.jpg'

        #############################
        #     JPEG形式の圧縮処理     #
        #############################
        # ファイル名を取得
        input = transform(torch.reshape(input,[3,416,416]))
        file_name = os.path.splitext(os.path.basename(jpeg_imgefile))[0]
        
        # バイナリモードファイルをPILイメージで取得
        im = input
        # JPEG形式の圧縮を実行
        im_io = BytesIO()
        im.save(im_io, 'JPEG', quality = COMPRESS_QUALITY)
        with open('comp_' + file_name + '.jpg', mode='wb') as outputfile:
            # 出力ファイル(comp_png_image.png)に書き込み
            outputfile.write(im_io.getvalue())
            fs += os.path.getsize('comp_' + file_name + '.jpg')

        im = Image.open('comp_' + file_name + '.jpg')

        input = torchvision.transforms.functional.to_tensor(im)
        
        from torchvision.utils import save_image
        save_image(input, "./masked_image_comp.jpg")

        return input, fs

    

class COCOEvaluator_vid:
    def __init__(self, dataset_dir, anno_path, img_size, batch_size):
        # MS COCO の評価を行うときに使用する閾値
        self.conf_threshold = 0.005
        self.nms_threshold = 0.45
        # Dataset を作成する。
        self.dataset = COCODataset(dataset_dir, anno_path, img_size=img_size)
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

    def output_to_dict_vid(self, output, img_id):
        detection = []
        for x1, y1, x2, y2, obj_conf, class_conf, label in output:

            # label = int(label.item()) + 1
            # COCO_label = [2,3,4,5,6,7,15,17,18,19,21,22,23]
            # VID_label = [1,3,4,5,6,7,9,11,15,19,22,26,30]

            label = int(label.item())
            COCO_label = [1,2,3,4,5,6,14,16,17,18,20,21,22]
            # VID_label = [0,2,3,4,5,6,8,10,14,18,21,25,29]
            VID_label = [3,6,18,0,5,25,4,8,14,21,10,2,29]

            if label in COCO_label:
            # if(np.any(label == COCO_label)):
                ind = COCO_label.index(label)
                vid_label = VID_label[ind]

                # vid_label -= 1

                # print(label, vid_label, ind)

                bbox = {
                    "image_id": int(img_id),
                    "category_id": self.dataset.category_ids[int(vid_label)],
                    # "category_id": self.dataset.category_ids[int(COCO_label)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(obj_conf * class_conf),
                }

                detection.append(bbox)

        return detection
    
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


    def evaluate(self, model):
        model.eval()
        device = next(model.parameters()).device

        detections = []
        for inputs, _, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]
            with torch.no_grad():
                # 順伝搬を行う。
                outputs = model(inputs)
                # 後処理を行う。
                outputs = postprocess(
                    outputs, self.conf_threshold, self.nms_threshold, pad_infos
                )

            for output, img_id in zip(outputs, img_ids):
                detections += self.output_to_dict_vid(output, img_id)

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        return ap50_95, ap50



class COCOEvaluator2(COCOEvaluator):
    def evaluate(self, model_edge, model_cloud):
        #推論モードにする
        model_edge.eval()
        model_cloud.eval()
        device = next(model_edge.parameters()).device

        detections = []

        count = 0

        self.up = torch.nn.Upsample(scale_factor=2)
        #比較するobjectnessの宣言
        #objectness_before_13 = torch.zeros(2535)
        #objectness_before_13 = objectness_before_13.to(device)
        #objectness_before_26 = torch.zeros(26, 26)
        

        for inputs, _, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]

            with torch.no_grad():
                #scales = [13, 26]
                #objectness = []
                #objectness_before = torch.zeros_like(objectness)

                # edgeの順伝搬を行う。(ここでfeature_edge_headにedgeの前半結果をいれる)
                outputs_edge, feature_edge_head = model_edge(inputs, usecloud=True)


                #今回のconfidenceの結果を代入
                confidence = outputs_edge[:, :, 4:5] * outputs_edge[:, :, 5:]

                #処理1のしきい値
                #conf_threshold = 0.5
                conf_threshold = 0.005
                #confidenceとしきい値の比較
                ind_1 = (confidence < conf_threshold)
                tmp = torch.reshape(ind_1, [-1, 80])
                tmp_ = torch.ones_like(tmp[:,0:1], dtype=torch.bool)
                for i in range(tmp.shape[1]):
                    tmp_*=tmp[:,i:i+1]
                tmp_ = torch.reshape(tmp_, [2535])

                output_1 = torch.clone(outputs_edge)
                output_1[0][tmp_] = 0

                output_1 = postprocess(
                    output_1, self.conf_threshold, self.nms_threshold, pad_infos
                            )
                

                if count == 0:
                    confidence_before = torch.zeros_like(confidence)
                    outputs_before = torch.zeros_like(outputs_edge)

                residual_conf = confidence_before - confidence

                residual_conf[0][~tmp_] = 100

                res_conf_threshold = 0.5
                ind_3 = (residual_conf < res_conf_threshold)

                tmp = torch.reshape(ind_3, [-1, 80])
                tmp_ = torch.ones_like(tmp[:,0:1], dtype=torch.bool)
                for i in range(tmp.shape[1]):
                    tmp_*=tmp[:,i:i+1]
                tmp_1 = torch.reshape(tmp_[:26*26*3], [3, 26*26])
                tmp_2 = torch.reshape(tmp_[26*26*3:], [3, 13*13])
                tmp_ = torch.reshape(tmp_, [2535])


                outputs_before[0][~tmp_] = 0
                output_4 = postprocess(
                    outputs_before, self.conf_threshold, self.nms_threshold, pad_infos
                            )



                # tmp = torch.reshape(ind_3, [-1, 80])
                tmp__ = torch.ones_like(tmp_1[0:1], dtype=torch.bool)
                for i in range(tmp_1.shape[0]):
                    tmp__*=tmp_1[i:i+1]
                tmp__ = torch.reshape(tmp__, [1, 1, 26, 26]).float()
                tmp__ = self.up(tmp__)
                tmp__ = (tmp__ > 0)


                tmp___ = torch.ones_like(tmp_2[0:1], dtype=torch.bool)
                for i in range(tmp_2.shape[0]):
                    tmp___*=tmp_2[i:i+1]
                tmp___ = torch.reshape(tmp___, [1, 1, 13, 13]).float()
                tmp___ = self.up(self.up(tmp___))
                tmp___ = (tmp___ > 0)

                ind_52 = torch.reshape(tmp__ * tmp___ , [52,52])
                feature_edge_head = torch.reshape(feature_edge_head, [1,52,52,256])
                feature_edge_head[0][ind_52] = 0
                feature_edge_head = torch.reshape(feature_edge_head, [1,256,52,52])

                #featureをcloud送信
                output_3 = model_cloud(feature_edge_head)
                
                output_3 = postprocess(
                output_3, self.conf_threshold, self.nms_threshold, pad_infos
                )


                for output, img_id in zip(output_1, img_ids):
                    detections += self.output_to_dict(output, img_id)

                for output, img_id in zip(output_3, img_ids):
                    detections += self.output_to_dict(output, img_id)

                for output, img_id in zip(output_4, img_ids):
                    detections += self.output_to_dict(output, img_id)


                

                # if (output[:, :, 5:] * output[:, :, 4:5] >= conf_threshold):
                #     outputs = postprocess(
                #     outputs_edge, self.conf_threshold, self.nms_threshold, pad_infos
                #             )
                
                # #セルごとにconfidenceの差を比較
                # else:
                #     #for k in range(2325):
                #     residual = confidence_before - confidence

                #     if 1 >=residual.max()> 0.5: #前回とのセルのconfidenceの差が大きいならば

                #         #featureをcloud送信
                #         outputs_cloud = model_cloud(feature_edge_head)
                #         confidence = outputs_cloud[:, :, 4:5] * outputs_cloud[:, :, 5:]
                        
                #         outputs = postprocess(
                #         outputs_cloud, self.conf_threshold, self.nms_threshold, pad_infos
                #         )
                #         #現フレームのcloud処理で得られたobjectnessを保存
                #         confidence = torch.zeros_like(confidence)
                #     else:
                #         outputs = outputs_before

                
            # for output, img_id in zip(outputs, img_ids):
            #     detections += self.output_to_dict(output, img_id)
            
            confidence_before = confidence
            outputs_before = outputs_edge

            count+=1

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        return ap50_95, ap50
    


                #     #for k in objectness:
            #     for k in range(len_objectness): 
            #         #セルごとに確認もし減少幅(しきい値)が大きい部分がある場合にはcloudでもう一回検証
            #         #if (objectness_before[k, 0].item() - objectness[k, 0].item() > 0.5):
            #         if (objectness_before[k, 0] - objectness[k, 0] > 1):
            #             #outputs_edge, feature_edge_head = model_edge(inputs, usecloud=True)
            #             #cloud側で処理
            #             outputs_cloud = model_cloud(feature_edge_head)

            #             #output, objectnessの値を更新
            #             outputs, objectness = postprocess(
            #             outputs_cloud, self.conf_threshold, self.nms_threshold, pad_infos
            #             )
            #             #現フレームのcloud処理で得られたobjectnessを保存
            #             #objectness_before = objectness
            #             break

            #         else: 
            #             continue
                
            #     #edge側もしくはcloud側で得られたobjectnessをobjectnessに代入
                # objectness_before = objectness
                
            #     # 後処理を行う。
            #     #outputs = postprocess(
            #     #    outputs_edge, self.conf_threshold, self.nms_threshold, pad_infos
            #     #)

            #     #順伝搬を行う。
            #     #outputs = model_edge(inputs)
            #     #outputs_edge, feature_edge_head = model_edge(inputs, usecloud=True)
            #     #outputs_cloud = model_cloud(feature_edge_head)
            #     # 後処理を行う。
            #     #outputs = postprocess(
            #     #    outputs_edge, self.conf_threshold, self.nms_threshold, pad_infos
            #     #)

            # #outputsにはedgeもしくはcloudで更新された値が入るはず



            #print(objectness.size())
                #objectness獲得
                # for scale in scales:
                    #scale_outputs = outputs_edge[scale]
                    #batch_size, grid_h, grid_w = scale_outputs.shape
                    #num_classes = scale_outputs.shape[1] - 5
                    #confidence = scale_outputs[..., 4].view(batch_size, num_classes, grid_h, grid_w)
                    #objectness.append(confidence)
                
                # #objectness_beforeの宣言
                # if (len(objectness_before) == 0):
                #     objectness_before = torch.zeros_like(objectness)
                #     #objectness_before = objectness_before.to(dtype=torch.long)
                #     objectness_before = objectness_before.detach()
                #     objectness_before = objectness_before.tolist()
                #     objectness_before = np.array(objectness_before)
                # #objectness_before = torch.zeros_like(objectness, dtype=torch.long)
                # #print(objectness)
                # #print(objectness.size())
                # #print(objectness)
                # #objectness = objectness.to(dtype=torch.long)

                # objectness = objectness.detach()
                # objectness = objectness.tolist()
                # objectness = np.array(objectness)

                # print("objectness", objectness)
                # len_objectness = len(objectness)
                # print(len_objectness)
                # print("objectness_before", objectness_before)




class COCOEvaluator3(COCOEvaluator):
    def evaluate(self, model_edge, model_cloud):
        #推論モードにする
        model_edge.eval()
        model_cloud.eval()
        device = next(model_edge.parameters()).device

        detections = []

        count = 0

        self.up = torch.nn.Upsample(scale_factor=2)
        #比較するobjectnessの宣言
        

        for inputs, _, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]

            with torch.no_grad():

                # edgeの順伝搬を行う。(ここでfeature_edge_headにedgeの前半結果をいれる)
                if count==0:
                    outputs_edge, feature_edge_head = model_edge(inputs, usecloud=True)

                    outputs_edge = postprocess(
                        outputs_edge, self.conf_threshold, self.nms_threshold, pad_infos
                        )

                    for output, img_id in zip(outputs_edge, img_ids):
                        det = self.output_to_dict(output, img_id)
                        detections += det

                    outputs_before = det

                else:
                    residual = inputs - before

                    # frame
                    thres1 = 0.00001
                    r = torch.abs(residual.sum())/torch.numel(residual)
                    if r < thres1:

                            detections += outputs_before


                    else:

                        outputs_edge, feature_edge_head = model_edge(inputs, usecloud=True)


                        #今回のconfidenceの結果を代入
                        confidence = outputs_edge[:, :, 4:5] * outputs_edge[:, :, 5:]

                        #処理1のしきい値
                        conf_threshold = 0.5
                        # conf_threshold = 0.005
                        #confidenceとしきい値の比較
                        ind_1 = (confidence < conf_threshold)
                        tmp = torch.reshape(ind_1, [-1, 80])
                        tmp_ = torch.ones_like(tmp[:,0:1], dtype=torch.bool)
                        for i in range(tmp.shape[1]):
                            tmp_*=tmp[:,i:i+1]
                        tmp_ = torch.reshape(tmp_, [2535])

                        output_1 = torch.clone(outputs_edge)
                        output_1[0][tmp_] = 0

                        output_1 = postprocess(
                            output_1, self.conf_threshold, self.nms_threshold, pad_infos
                                    )
                        
                        tmp_1 = torch.reshape(tmp_[:26*26*3], [3, 26*26])
                        tmp_2 = torch.reshape(tmp_[26*26*3:], [3, 13*13])
                        tmp_ = torch.reshape(tmp_, [2535])


                        # tmp = torch.reshape(ind_3, [-1, 80])
                        tmp__ = torch.ones_like(tmp_1[0:1], dtype=torch.bool)
                        for i in range(tmp_1.shape[0]):
                            tmp__*=tmp_1[i:i+1]
                        tmp__ = torch.reshape(tmp__, [1, 1, 26, 26]).float()
                        tmp__ = self.up(tmp__)
                        tmp__ = (tmp__ > 0)


                        tmp___ = torch.ones_like(tmp_2[0:1], dtype=torch.bool)
                        for i in range(tmp_2.shape[0]):
                            tmp___*=tmp_2[i:i+1]
                        tmp___ = torch.reshape(tmp___, [1, 1, 13, 13]).float()
                        tmp___ = self.up(self.up(tmp___))
                        tmp___ = (tmp___ > 0)

                        ind_52 = torch.reshape(tmp__ * tmp___ , [52,52])
                        feature_edge_head = torch.reshape(feature_edge_head, [1,52,52,256])
                        feature_edge_head[0][ind_52] = 0
                        feature_edge_head = torch.reshape(feature_edge_head, [1,256,52,52])

                        #featureをcloud送信
                        output_3 = model_cloud(feature_edge_head)
                        
                        output_3 = postprocess(
                        output_3, self.conf_threshold, self.nms_threshold, pad_infos
                        )


                        for output, img_id in zip(output_1, img_ids):
                            det1 = self.output_to_dict(output, img_id)
                            detections += det1

                        for output, img_id in zip(output_3, img_ids):
                            det3 = self.output_to_dict(output, img_id)
                            detections += det3

                        outputs_before = det1 + det3


            before = inputs

            count+=1

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

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
    # コンフィグ
    COMPRESS_QUALITY = qa # 圧縮のクオリティ

    # JPEG形式とPNG形式の画像ファイルを用意
    jpeg_imgefile = 'jpeg_image.jpg'

    #############################
    #     JPEG形式の圧縮処理     #
    #############################
    # ファイル名を取得
    input = transform(torch.reshape(input,[3,416,416]))
    file_name = os.path.splitext(os.path.basename(jpeg_imgefile))[0]
    
    # バイナリモードファイルをPILイメージで取得
    im = input
    # JPEG形式の圧縮を実行
    im_io = BytesIO()
    im.save(im_io, 'JPEG', quality = COMPRESS_QUALITY)
    with open('comp_' + file_name + '.jpg', mode='wb') as outputfile:
        # 出力ファイル(comp_png_image.png)に書き込み
        outputfile.write(im_io.getvalue())
        fs += os.path.getsize('comp_' + file_name + '.jpg')

    im = Image.open('comp_' + file_name + '.jpg')

    input = torchvision.transforms.functional.to_tensor(im)
    
    from torchvision.utils import save_image
    save_image(input, "./masked_image_comp.jpg")

    return input, fs


def compress_and_calculate_bpp(tensor, quality=85):
    from PIL import Image
    import io
    # PyTorch TensorをPIL Imageに変換
    tensor = tensor.clamp(0, 1)  # テンソルの値を0から1に制約
    image = tensor.mul(255).byte()  # 0から1の範囲を0から255に変換
    image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())

    # 画像を一時的なバッファに保存
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)

    # 画像のビット深度（bpp）を計算
    bpp = (8 * len(buffer.getvalue())) / (image.width * image.height)

    return bpp



class COCOEvaluator4(COCOEvaluator): #edge-cloud by confidence
    def evaluate(self, model_edge, model_cloud, q, thres1):
        #推論モードにする
        model_edge.eval()
        model_cloud.eval()
        device = next(model_edge.parameters()).device

        detections = []

        count = 0

        fs = 0

        self.up = torch.nn.Upsample(scale_factor=2)
        #比較するobjectnessの宣言
        

        for inputs, _, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]

            with torch.no_grad():

                # edgeの順伝搬を行う。(ここでfeature_edge_headにedgeの前半結果をいれる)
                if count==0:
                    outputs = model_cloud(inputs)

                    outputs = postprocess(
                        outputs, self.conf_threshold, self.nms_threshold, pad_infos
                        )

                    for output, img_id in zip(outputs, img_ids):
                        det = self.output_to_dict(output, img_id)
                        detections += det

                    outputs_before = det

                    key=torch.clone(inputs).detach()

                else:
                    residual = inputs - key
                    

                    # frame
                    #thres1 = 0.005
                    # thres1 = 0.000000001
                    # thres1 = 1/256
                    r = torch.abs(residual).sum()/torch.numel(residual)
                    if r < thres1:

                            detections += outputs_before

                    else:

                        key=torch.clone(inputs).detach()

                        outputs_edge = model_edge(inputs)


                        #今回のconfidenceの結果を代入
                        confidence = outputs_edge[:, :, 4:5] * outputs_edge[:, :, 5:]

                        #処理1のしきい値
                        #conf_threshold = 0.5
                        conf_threshold = 0.65
                        #conf_threshold = 0.005
                        #confidenceとしきい値の比較
                        ind_1 = (confidence < conf_threshold)
                        tmp = torch.reshape(ind_1, [-1, 80])
                        tmp_ = torch.ones_like(tmp[:,0:1], dtype=torch.bool)
                        for i in range(tmp.shape[1]):
                            tmp_*=tmp[:,i:i+1]
                        tmp_ = torch.reshape(tmp_, [2535])

                        output_1 = torch.clone(outputs_edge)
                        output_1[0][tmp_] = 0

                        output_1 = postprocess(
                            output_1, self.conf_threshold, self.nms_threshold, pad_infos
                                    )
                        
                        ##############################背景マスク
                        conf_threshold = 0.001
                        # conf_threshold = 0.005
                        #confidenceとしきい値の比較
                        ind_1 = (confidence < conf_threshold)
                        tmp = torch.reshape(ind_1, [-1, 80])
                        tmp_ = torch.ones_like(tmp[:,0:1], dtype=torch.bool)
                        for i in range(tmp.shape[1]):
                            tmp_*=tmp[:,i:i+1]
                        tmp_ = torch.reshape(tmp_, [2535])

                        tmp_1 = torch.reshape(tmp_[13*13*3:], [3, 26*26])
                        tmp_2 = torch.reshape(tmp_[:13*13*3], [3, 13*13])
                        tmp_ = torch.reshape(tmp_, [2535])


                        # from torchvision.utils import save_image
                        # save_image(inputs, "masked_images.jpg")
                        # mask = torch.reshape(tmp_1.float(), [3,1,26,26])
                        # mask2 = torch.reshape(tmp_2.float(), [3,1,13,13])
                        # save_image(mask, "mask26.jpg")
                        # save_image(mask2, "mask13.jpg")


                        # tmp = torch.reshape(ind_3, [-1, 80])
                        tmp__ = torch.ones_like(tmp_1[0:1], dtype=torch.bool)
                        for i in range(tmp_1.shape[0]):
                            tmp__*=tmp_1[i:i+1]
                        tmp__ = torch.reshape(tmp__, [1, 1, 26, 26]).float()
                        tmp__ = self.up(self.up(self.up(self.up(tmp__))))
                        tmp__ = (tmp__ > 0)


                        tmp___ = torch.ones_like(tmp_2[0:1], dtype=torch.bool)
                        for i in range(tmp_2.shape[0]):
                            tmp___*=tmp_2[i:i+1]
                        tmp___ = torch.reshape(tmp___, [1, 1, 13, 13]).float()
                        tmp___ = self.up(self.up(self.up(self.up(self.up(tmp___)))))
                        tmp___ = (tmp___ > 0)

                        
                        ind_52 = torch.reshape(tmp__ * tmp___ , [1,416,416])
                        ind_52 = ind_52
                        ind_52_ = torch.zeros_like(inputs)
                        ind_52_[0,0] = ind_52
                        ind_52_[0,1] = ind_52
                        ind_52_[0,2] = ind_52
                        # inputs = torch.reshape(inputs, [1,416,416,3])
                        inputs[ind_52_.bool()] = 0.5 #
                        # inputs = torch.reshape(inputs, [1,3,416,416])

                        from torchvision.utils import save_image
                        save_image(inputs, "masked_images.jpg")
                        # mask = torch.reshape(ind_52, [1,1,416,416])
                        save_image(ind_52.float(), "mask.jpg")
                        #############################
                        
                        for output, img_id in zip(output_1, img_ids):
                            det1 = self.output_to_dict(output, img_id)
                            detections += det1
                        

                        bbs = []
                        for d in det1:
                            x = encode_bboxes(d["bbox"], pad_infos)
                            x = torch.from_numpy(x.astype(np.float32)).clone()
                            bbs.append(x)

                        for b in bbs:
                            # inputs[:,:, int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] = 0
                            inputs[:,:, int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] = 0.5
                        
                        from torchvision.utils import save_image
                        save_image(inputs, "./masked_image.jpg")

                        inputs, fs = compression(inputs, fs, qa = q)
                        inputs = torch.reshape(inputs, [1,3,416,416]).to(device)

                        #featureをcloud送信
                        output_3 = model_cloud(inputs)
                        
                        output_3 = postprocess(
                        output_3, self.conf_threshold, self.nms_threshold, pad_infos
                        )
                        

                        for output, img_id in zip(output_3, img_ids):
                            det3 = self.output_to_dict(output, img_id)
                            detections += det3

                        outputs_before = det1 + det3
                        # from torchvision.utils import save_image
                        # save_image(outputs_before, "./final_masked_image.jpg")
                        #outputs_before =  det3


            count+=1

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        print("File size", fs)

        return ap50_95, ap50, fs
    

class COCOEvaluator5(COCOEvaluator): #v3のみ
    def evaluate(self, model_edge, model_cloud, q):
        #推論モードにする
        model_edge.eval()
        model_cloud.eval()
        device = next(model_edge.parameters()).device

        detections = []

        count = 0

        fs = 0

        self.up = torch.nn.Upsample(scale_factor=2)
        #比較するobjectnessの宣言
        

        for inputs, _, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]

            with torch.no_grad():

                    inputs, fs = compression(inputs, fs, qa=q)
                    # 圧縮とbpp計算
                    bpp_result = compress_and_calculate_bpp(inputs)  
                    inputs = torch.reshape(inputs, [1,3,416,416]).to(device)

                    outputs = model_cloud(inputs)
                    #outputs = model_edge(inputs)

                
                    outputs = postprocess(
                        outputs, self.conf_threshold, self.nms_threshold, pad_infos
                        )

                    for output, img_id in zip(outputs, img_ids):
                        det = self.output_to_dict(output, img_id)
                        detections += det

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        print("FIle size", fs)
        
        
        return ap50_95, ap50, fs
    
class COCOEvaluator6(COCOEvaluator): #objectness version
    def evaluate(self, model_edge, model_cloud, q, thres1):
        #推論モードにする
        model_edge.eval()
        model_cloud.eval()
        device = next(model_edge.parameters()).device

        detections = []

        count = 0

        fs = 0

        self.up = torch.nn.Upsample(scale_factor=2)
        #比較するobjectnessの宣言
        

        for inputs, _, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]

            with torch.no_grad():

                # edgeの順伝搬を行う。(ここでfeature_edge_headにedgeの前半結果をいれる)
                if count==0:
                    outputs = model_cloud(inputs)

                    outputs = postprocess(
                        outputs, self.conf_threshold, self.nms_threshold, pad_infos
                        )

                    for output, img_id in zip(outputs, img_ids):
                        det = self.output_to_dict(output, img_id)
                        detections += det

                    outputs_before = det

                    key=torch.clone(inputs).detach()

                else:
                    residual = inputs - key
                    

                    # frame
                    #thres1 = 0.005
                    # thres1 = 0.000000001
                    # thres1 = 1/256
                    r = torch.abs(residual).sum()/torch.numel(residual)
                    if r < thres1:

                            detections += outputs_before

                    else:

                        key=torch.clone(inputs).detach()

                        outputs_edge = model_edge(inputs)


                        #今回のconfidenceの結果を代入
                        # confidence = outputs_edge[:, :, 4:5] * outputs_edge[:, :, 5:]

                        objectness = outputs_edge[:, :, 4:5]
                        confidence = objectness 

                        #処理1のしきい値
                        #conf_threshold = 0.5
                        conf_threshold = 0.5
                        #conf_threshold = 0.005
                        #confidenceとしきい値の比較
                        ind_1 = (confidence < conf_threshold)
                        tmp = torch.reshape(ind_1, [-1, 1])
                        tmp_ = torch.ones_like(tmp[:,0:1], dtype=torch.bool)
                        for i in range(tmp.shape[1]):
                            tmp_*=tmp[:,i:i+1]
                        tmp_ = torch.reshape(tmp_, [2535])

                        output_1 = torch.clone(outputs_edge)
                        output_1[0][tmp_] = 0

                        output_1 = postprocess(
                            output_1, self.conf_threshold, self.nms_threshold, pad_infos
                                    )
                        
                        ##############################背景マスク
                        conf_threshold = 0.001
                        # conf_threshold = 0.005
                        #confidenceとしきい値の比較
                        ind_1 = (confidence < conf_threshold)
                        tmp = torch.reshape(ind_1, [-1, 1])
                        tmp_ = torch.ones_like(tmp[:,0:1], dtype=torch.bool)
                        for i in range(tmp.shape[1]):
                            tmp_*=tmp[:,i:i+1]
                        tmp_ = torch.reshape(tmp_, [2535])

                        tmp_1 = torch.reshape(tmp_[13*13*3:], [3, 26*26])
                        tmp_2 = torch.reshape(tmp_[:13*13*3], [3, 13*13])
                        tmp_ = torch.reshape(tmp_, [2535])


                        # from torchvision.utils import save_image
                        # save_image(inputs, "masked_images.jpg")
                        # mask = torch.reshape(tmp_1.float(), [3,1,26,26])
                        # mask2 = torch.reshape(tmp_2.float(), [3,1,13,13])
                        # save_image(mask, "mask26.jpg")
                        # save_image(mask2, "mask13.jpg")


                        # tmp = torch.reshape(ind_3, [-1, 80])
                        tmp__ = torch.ones_like(tmp_1[0:1], dtype=torch.bool)
                        for i in range(tmp_1.shape[0]):
                            tmp__*=tmp_1[i:i+1]
                        tmp__ = torch.reshape(tmp__, [1, 1, 26, 26]).float()
                        tmp__ = self.up(self.up(self.up(self.up(tmp__))))
                        tmp__ = (tmp__ > 0)


                        tmp___ = torch.ones_like(tmp_2[0:1], dtype=torch.bool)
                        for i in range(tmp_2.shape[0]):
                            tmp___*=tmp_2[i:i+1]
                        tmp___ = torch.reshape(tmp___, [1, 1, 13, 13]).float()
                        tmp___ = self.up(self.up(self.up(self.up(self.up(tmp___)))))
                        tmp___ = (tmp___ > 0)

                        
                        ind_52 = torch.reshape(tmp__ * tmp___ , [1,416,416])
                        ind_52 = ind_52
                        ind_52_ = torch.zeros_like(inputs)
                        ind_52_[0,0] = ind_52
                        ind_52_[0,1] = ind_52
                        ind_52_[0,2] = ind_52
                        # inputs = torch.reshape(inputs, [1,416,416,3])
                        inputs[ind_52_.bool()] = 0.5 #
                        # inputs = torch.reshape(inputs, [1,3,416,416])

                        from torchvision.utils import save_image
                        save_image(inputs, "masked_images.jpg")
                        # mask = torch.reshape(ind_52, [1,1,416,416])
                        save_image(ind_52.float(), "mask.jpg")
                        #############################
                        
                        for output, img_id in zip(output_1, img_ids):
                            det1 = self.output_to_dict(output, img_id)
                            detections += det1
                        

                        bbs = []
                        for d in det1:
                            x = encode_bboxes(d["bbox"], pad_infos)
                            x = torch.from_numpy(x.astype(np.float32)).clone()
                            bbs.append(x)

                        for b in bbs:
                            # inputs[:,:, int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] = 0
                            inputs[:,:, int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])] = 0.5
                        
                        from torchvision.utils import save_image
                        save_image(inputs, "./masked_image.jpg")

                        inputs, fs = compression(inputs, fs, qa = q)
                        inputs = torch.reshape(inputs, [1,3,416,416]).to(device)

                        #featureをcloud送信
                        output_3 = model_cloud(inputs)
                        
                        output_3 = postprocess(
                        output_3, self.conf_threshold, self.nms_threshold, pad_infos
                        )
                        

                        for output, img_id in zip(output_3, img_ids):
                            det3 = self.output_to_dict(output, img_id)
                            detections += det3

                        outputs_before = det1 + det3
                        #outputs_before =  det3


            count+=1

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        print("File size", fs)

        return ap50_95, ap50, fs



def encode_bboxes(bboxes, pad_info):
    scale_x, scale_y, dx, dy = pad_info

    bboxes *= np.array([scale_x.item(), scale_y.item(), scale_x.item(), scale_y.item()])
    bboxes[0] += dx.item()
    bboxes[1] += dy.item()

    return bboxes