from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import (
    FasterRCNN, FasterRCNN_ResNet50_FPN_V2_Weights, FCOS_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, fcos_resnet50_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.ops import FrozenBatchNorm2d, MultiScaleRoIAlign


class TinyBackbone(nn.Module):
    """Only for integration smoke tests; never a paper model."""
    out_channels = 32

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())

    def forward(self, x):
        return OrderedDict([("0", self.net(x))])


class GeometryDetector(nn.Module):
    """Faster R-CNN + training-only quadrilateral geometry supervision on GT RoIs.

    Backbone receives auxiliary gradients. Detection inference is unchanged.
    Orientation is represented modulo pi by sin(2 theta), cos(2 theta).
    Square-like enclosures do not have a trustworthy principal direction.
    """

    def __init__(self, detector, geometry_mode="none", geometry_weight=0.2, crack_only=True):
        super().__init__()
        if geometry_mode not in {"none", "aspect", "full"}:
            raise ValueError("geometry_mode must be none, aspect or full")
        self.detector = detector
        self.geometry_mode, self.geometry_weight, self.crack_only = geometry_mode, geometry_weight, crack_only
        # Creating this AFTER the detector ensures seed-matched detection weights.
        if geometry_mode != "none":
            channels = detector.backbone.out_channels
            self.geometry_head = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(),
                                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 3))

    def forward(self, images, targets=None):
        if not self.training or self.geometry_mode == "none":
            return self.detector(images, targets)
        if targets is None:
            raise ValueError("Training requires targets")
        images, targets = self.detector.transform(images, targets)
        for target in targets:
            if (target["boxes"][:, 2:] <= target["boxes"][:, :2]).any():
                raise ValueError("Degenerate training boxes")
        features = self.detector.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, rpn_losses = self.detector.rpn(images, features, targets)
        _, detection_losses = self.detector.roi_heads(features, proposals, images.image_sizes, targets)
        masks = [t["geometry_valid"] & ((t["labels"] == 2) if self.crack_only else
                                         torch.ones_like(t["geometry_valid"])) for t in targets]
        gt_boxes = [t["boxes"][m] for t, m in zip(targets, masks)]
        losses = {**rpn_losses, **detection_losses}
        if not sum(len(b) for b in gt_boxes):
            # Keep all parameters in the graph, even with no eligible annotations.
            zero = sum(p.sum()*0 for p in self.geometry_head.parameters())
            losses["loss_geometry_aspect"] = zero
            if self.geometry_mode == "full":
                losses["loss_geometry_orientation"] = zero
            return losses
        pooled = self.detector.roi_heads.box_roi_pool(features, gt_boxes, images.image_sizes)
        predicted = self.geometry_head(pooled).float()
        truth = torch.cat([t["geometry"][m] for t, m in zip(targets, masks)]).float()
        reliability = torch.cat([t["anisotropy"][m] for t, m in zip(targets, masks)]).float()
        losses["loss_geometry_aspect"] = self.geometry_weight * F.smooth_l1_loss(predicted[:, 0], truth[:, 0])
        if self.geometry_mode == "full":
            # Anisotropy continuously downweights unstable orientations.
            orientation = F.smooth_l1_loss(predicted[:, 1:], truth[:, 1:], reduction="none").mean(1)
            losses["loss_geometry_orientation"] = self.geometry_weight * (orientation * reliability).mean()
        return losses


def build_model(config, pretrained=None, model_spec=None):
    if model_spec is not None:
        import copy
        config = copy.deepcopy(config)
        config["model"]["hf_config"] = model_spec
    cfg = config["model"]
    pretrained = cfg["pretrained"] if pretrained is None else pretrained
    kwargs = dict(min_size=cfg["min_size"], max_size=cfg["max_size"],
                  box_score_thresh=cfg.get("score_floor", 0.001),
                  box_detections_per_img=100)
    architecture = cfg["architecture"]
    if architecture == "yolo11s":
        from .adapters import YOLODetector
        return YOLODetector(config, pretrained)
    if architecture == "rtdetr_v2_r50vd":
        from .adapters import RTDETRv2Detector
        return RTDETRv2Detector(config, pretrained)
    if architecture == "fasterrcnn_r50_fpn_v2":
        # Build identical architecture/freezing for first training, resume and evaluation.
        # Passing weights=None directly to the builder otherwise changes frozen layers.
        detector = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kwargs)
        if pretrained:
            detector.load_state_dict(FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict(progress=True, check_hash=True))
        freeze_backbone(detector, cfg.get("trainable_backbone_layers", 3))
        detector.roi_heads.box_predictor = FastRCNNPredictor(detector.roi_heads.box_predictor.cls_score.in_features, 4)
    elif architecture == "tiny_smoke":
        detector = FasterRCNN(TinyBackbone(), num_classes=4,
                             rpn_anchor_generator=AnchorGenerator(((8, 16, 32, 64),), ((0.5, 1., 2.),)),
                             box_roi_pool=MultiScaleRoIAlign(["0"], output_size=3, sampling_ratio=2),
                             rpn_pre_nms_top_n_train=100, rpn_post_nms_top_n_train=50,
                             rpn_pre_nms_top_n_test=100, rpn_post_nms_top_n_test=50,
                             box_batch_size_per_image=32, **kwargs)
    elif architecture == "fcos_r50_fpn":
        if cfg["geometry_mode"] != "none":
            raise ValueError("FCOS is a baseline; geometry head is implemented on Faster R-CNN only")
        detector = fcos_resnet50_fpn(weights=None, weights_backbone=None,
                                   min_size=cfg["min_size"], max_size=cfg["max_size"],
                                   score_thresh=cfg.get("score_floor", 0.001), detections_per_img=100)
        # Official pretrained FCOS uses FrozenBatchNorm; retain it on reload even
        # when downloading pretrained weights is disabled by the caller.
        if cfg["pretrained"]:
            convert_frozen_batchnorm(detector.backbone)
        if pretrained:
            detector.load_state_dict(FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True, check_hash=True))
        freeze_backbone(detector, cfg.get("trainable_backbone_layers", 3))
        detector.head.classification_head = FCOSClassificationHead(detector.backbone.out_channels,
                                                                  detector.anchor_generator.num_anchors_per_location()[0], 4)
        return detector
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return GeometryDetector(detector, cfg["geometry_mode"], cfg["geometry_weight"], cfg.get("crack_only", True))


def freeze_backbone(detector, trainable_layers):
    if trainable_layers not in range(6):
        raise ValueError("trainable_backbone_layers must be between 0 and 5")
    names = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        names.append("bn1")
    for name, parameter in detector.backbone.body.named_parameters():
        parameter.requires_grad_(any(name.startswith(prefix) for prefix in names))


def convert_frozen_batchnorm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                frozen.weight.copy_(child.weight)
                frozen.bias.copy_(child.bias)
                frozen.running_mean.copy_(child.running_mean)
                frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            convert_frozen_batchnorm(child)
