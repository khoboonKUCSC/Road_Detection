"""Modern detectors with the same loss/output contract as the torchvision models.

Dependencies are lazy so the original models and CPU smoke tests work offline.
No native-library validation metrics are used: all predictions go to COCOeval.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def letterbox(images, targets, size):
    tensors, transformed, metadata = [], [], []
    for i, image in enumerate(images):
        height, width = image.shape[-2:]
        gain = min(size/height, size/width)
        nh, nw = round(height*gain), round(width*gain)
        resized = F.interpolate(image[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
        left, top = (size-nw)//2, (size-nh)//2
        tensors.append(F.pad(resized, (left, size-nw-left, top, size-nh-top), value=114/255))
        # Rounding means x and y gains can differ slightly.
        gx, gy = nw/width, nh/height
        metadata.append((height, width, gx, gy, left, top))
        if targets is not None:
            boxes = targets[i]["boxes"].clone()
            boxes *= boxes.new_tensor([gx, gy, gx, gy])
            boxes += boxes.new_tensor([left, top, left, top])
            cxcy = (boxes[:, :2]+boxes[:, 2:])/2/size
            wh = (boxes[:, 2:]-boxes[:, :2])/size
            transformed.append(torch.cat([cxcy, wh], 1))
    return torch.stack(tensors), transformed, metadata


class YOLODetector(nn.Module):
    def __init__(self, config, pretrained):
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel
        cfg = config["model"]
        self.network = DetectionModel(cfg.get("yolo_yaml", "yolo11s.yaml"), ch=3, nc=3, verbose=False)
        self.size = cfg.get("input_size", 640)
        self.floor = cfg.get("score_floor", .001)
        self.nms_iou = cfg.get("nms_iou", .7)
        # Native loss reads these hyperparameters, normally set by native trainer.
        from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
        self.network.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "box": 7.5, "cls": .5, "dfl": 1.5})
        if pretrained:
            from ultralytics import YOLO
            self.network.load(YOLO(cfg.get("weights_id", "yolo11s.pt")).model, verbose=False)

    def forward(self, images, targets=None):
        pixels, boxes, metadata = letterbox(images, targets, self.size)
        if self.training:
            if targets is None:
                raise ValueError("YOLO training requires targets")
            batch = {"img": pixels, "batch_idx": torch.cat([torch.full((len(t["labels"]),), i, device=pixels.device)
                                                           for i, t in enumerate(targets)]),
                     "cls": torch.cat([t["labels"]-1 for t in targets]).float().reshape(-1, 1),
                     "bboxes": torch.cat(boxes)}
            losses, _ = self.network.loss(batch)
            # Ultralytics loss components are scaled by batch size.
            losses = losses.reshape(-1) / len(images)
            if len(losses) != 3:
                raise RuntimeError("Unexpected Ultralytics loss contract; use the pinned dependency version")
            return dict(zip(["loss_yolo_box", "loss_yolo_cls", "loss_yolo_dfl"], losses))
        from ultralytics.utils.ops import non_max_suppression
        decoded = self.network(pixels)
        outputs = non_max_suppression(decoded, conf_thres=self.floor, iou_thres=self.nms_iou, nc=3, max_det=100)
        results = []
        for result, (height, width, gx, gy, left, top) in zip(outputs, metadata):
            restored = result[:, :4].clone()
            restored -= restored.new_tensor([left, top, left, top])
            restored /= restored.new_tensor([gx, gy, gx, gy])
            restored[:, 0::2].clamp_(0, width)
            restored[:, 1::2].clamp_(0, height)
            valid = (restored[:, 2:] > restored[:, :2]).all(1)
            results.append({"boxes": restored[valid], "scores": result[valid, 4], "labels": result[valid, 5].long()+1})
        return results


class RTDETRv2Detector(nn.Module):
    def __init__(self, config, pretrained):
        super().__init__()
        from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection
        cfg = config["model"]
        self.size, self.floor = cfg.get("input_size", 640), cfg.get("score_floor", .001)
        self.processor = RTDetrImageProcessor(size={"height": self.size, "width": self.size},
                                             do_rescale=False, do_normalize=False, do_pad=False)
        identity = cfg.get("weights_id", "PekingU/rtdetr_v2_r50vd")
        labels = {0: "pothole", 1: "crack", 2: "manhole"}
        if pretrained:
            self.network = RTDetrV2ForObjectDetection.from_pretrained(identity, num_labels=3,
                              id2label=labels, label2id={v: k for k, v in labels.items()}, ignore_mismatched_sizes=True)
        else:
            model_config = RTDetrV2Config(num_labels=3, id2label=labels, label2id={v: k for k, v in labels.items()},
                                         anchor_image_size=[self.size, self.size])
            if cfg.get("hf_config"):
                model_config = RTDetrV2Config.from_dict(cfg["hf_config"])
            self.network = RTDetrV2ForObjectDetection(model_config)

    def forward(self, images, targets=None):
        annotations = None
        if self.training:
            if targets is None:
                raise ValueError("RT-DETRv2 training requires targets")
            annotations = []
            for target in targets:
                entries = []
                for box, label in zip(target["boxes"].detach().cpu().tolist(), target["labels"].detach().cpu().tolist()):
                    x1, y1, x2, y2 = box
                    entries.append({"bbox": [x1, y1, x2-x1, y2-y1], "category_id": label-1,
                                    "area": (x2-x1)*(y2-y1), "iscrowd": 0})
                annotations.append({"image_id": int(target["image_id"]), "annotations": entries})
        inputs = self.processor(images=[image.detach().cpu() for image in images], annotations=annotations, return_tensors="pt")
        device = next(self.network.parameters()).device
        inputs = inputs.to(device)
        if "labels" in inputs:
            inputs["labels"] = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in label.items()}
                                for label in inputs["labels"]]
        outputs = self.network(**inputs)
        if self.training:
            # HF outputs.loss is already the weighted sum, including auxiliary layers.
            return {"loss_rtdetr_total": outputs.loss}
        sizes = torch.tensor([im.shape[-2:] for im in images], device=device)
        processed = self.processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=self.floor)
        results = []
        for result, image in zip(processed, images):
            order = result["scores"].argsort(descending=True)[:100]
            boxes = result["boxes"][order].clone()
            height, width = image.shape[-2:]
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
            valid = (boxes[:, 2:] > boxes[:, :2]).all(1)
            results.append({"boxes": boxes[valid], "scores": result["scores"][order][valid],
                            "labels": result["labels"][order][valid]+1})
        return results
