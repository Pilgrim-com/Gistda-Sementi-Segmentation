# Models/DeepLabV3_MTL_Adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet50, deeplabv3_resnet101
)

class DeepLabV3_MTL_Adapter(nn.Module):
    """
    Multi-task wrapper around torchvision DeepLabV3:
      - Task 0 (road): uses DeepLabV3 main head -> n_road_classes
      - Task 1 (orientation): simple 1x1 head sitting on DeepLab 'out' features
    Returns:
      [ [road_logits_up], [angle_logits_up] ]
    (Lists so your loop that sums over scales keeps working.)
    """
    def __init__(
        self,
        n_road_classes=2,
        n_orient_classes=37,
        backbone='resnet101',           # 'resnet50'|'resnet101'
        pretrained_backbone=False,      # True: load imagenet on backbone
        output_stride=16                # 8 or 16 (16 by default in torchvision)
    ):
        super().__init__()

        # Build DeepLab with pretrained backbone if requested
        if backbone == 'resnet101':
            if pretrained_backbone:
                # Load pretrained model first, then modify num_classes
                from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
                self.deeplab = deeplabv3_resnet101(
                    weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
                    num_classes=21,  # default COCO classes
                    aux_loss=None
                )
                # Replace classifier head with correct number of classes
                self.deeplab.classifier[4] = nn.Conv2d(256, n_road_classes, kernel_size=1)
                print(f"[INFO] Loaded pretrained DeepLabV3-ResNet101 (COCO+VOC), replaced head with {n_road_classes} classes")
            else:
                self.deeplab = deeplabv3_resnet101(
                    weights=None,
                    num_classes=n_road_classes,
                    aux_loss=None
                )
        else:
            if pretrained_backbone:
                from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
                self.deeplab = deeplabv3_resnet50(
                    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
                    num_classes=21,
                    aux_loss=None
                )
                self.deeplab.classifier[4] = nn.Conv2d(256, n_road_classes, kernel_size=1)
                print(f"[INFO] Loaded pretrained DeepLabV3-ResNet50 (COCO+VOC), replaced head with {n_road_classes} classes")
            else:
                self.deeplab = deeplabv3_resnet50(
                    weights=None,
                    num_classes=n_road_classes,
                    aux_loss=None
                )

        # Orientation head: apply on road logits feature map or the last features
        # Here we keep it minimal: run 1x1 on the road logits, then upsample.
        # If you want richer features, grab decoder ASPP features instead.
        self.orient_head = nn.Conv2d(n_road_classes, n_orient_classes, kernel_size=1)

        self.n_road_classes = n_road_classes
        self.n_orient_classes = n_orient_classes

    def forward(self, x):
        # torchvision deeplab returns dict: {'out': BxCrxhxw, 'aux': ...}
        out = self.deeplab(x)
        road_logits = out['out']                                  # (B, n_road, h, w)

        # Orientation head on same (coarse) map
        angle_logits = self.orient_head(road_logits)              # (B, n_orient, h, w)

        # Upsample both to input size so your losses/metrics match labels
        size = x.shape[-2:]
        road_logits_up   = F.interpolate(road_logits,   size=size, mode='bilinear', align_corners=False)
        angle_logits_up  = F.interpolate(angle_logits,  size=size, mode='bilinear', align_corners=False)

        # Match your loop signature: lists of (possibly multi-scale) preds
        return [road_logits_up], [angle_logits_up]
