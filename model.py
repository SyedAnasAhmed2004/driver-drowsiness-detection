"""
Model definitions for Driver Distraction Detection.
Supports MobileNetV2 (lightweight, mobile-friendly) and ResNet50 (higher accuracy).
"""

import torch
import torch.nn as nn
from torchvision import models
from config import CONFIG


class DistractionDetector(nn.Module):
    """
    Transfer learning wrapper.
    - Loads a pretrained ImageNet backbone
    - Replaces the classifier head for 10-class distraction detection
    - Stage 1: only the new head is trainable (feature extraction)
    - Call unfreeze_backbone() for Stage 2 fine-tuning
    """

    def __init__(self, num_classes: int = 10, backbone: str = None, dropout: float = 0.4):
        super().__init__()
        backbone = backbone or CONFIG["backbone"]

        if backbone == "mobilenetv2":
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_features = base.classifier[1].in_features
            # Freeze all backbone layers initially
            for param in base.features.parameters():
                param.requires_grad = False
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(512, num_classes)
            )
            self.model = base

        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = base.fc.in_features
            for param in base.parameters():
                param.requires_grad = False
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(512, num_classes)
            )
            self.model = base

        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose 'mobilenetv2' or 'resnet50'.")

        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = 20):
        """
        Call after initial training to fine-tune the last N layers of the backbone.
        Lower learning rate recommended after calling this (e.g. 1e-5).
        """
        all_params = list(self.model.parameters())
        for param in all_params[-unfreeze_last_n_layers:]:
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] Unfrozen last {unfreeze_last_n_layers} layers. "
              f"Trainable params: {trainable:,}")

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick sanity check ────────────────────────────────────────────────────
if __name__ == "__main__":
    for bb in ["mobilenetv2", "resnet50"]:
        m = DistractionDetector(num_classes=10, backbone=bb)
        dummy = torch.randn(2, 3, 224, 224)
        out = m(dummy)
        print(f"{bb}: output shape = {out.shape} | "
              f"trainable params = {m.get_num_trainable_params():,}")
