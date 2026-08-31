"""Loss functions for training. Section 2.3's Dice+CE (standard) and
Dice+BCE (region-based training ablation), Section 2.4's batch-Dice vs.
sample-Dice (BD ablation), and a deep-supervision wrapper that combines
losses across the model's multiple output resolutions.

IMPORTANT — probabilities, not logits: NNUNet3D's SegmentationHead
already applies softmax/sigmoid internally (Fig. 1's "1x1x1 conv -
softmax" block), so `pred` here is always a probability tensor — never
raw logits. This is why the loss functions use NLL-style / plain BCE
math rather than nn.CrossEntropyLoss/nn.BCEWithLogitsLoss.
"""
from typing import List

import torch
import torch.nn as nn

EPS = 1e-5


def labels_to_regions(target: torch.Tensor) -> torch.Tensor:
    """Converts contiguous class-index labels {0,1,2,3} (background, NCR,
    ED, ET) into the 3 partially-overlapping BraTS evaluation regions:
    whole tumor (any of 1,2,3), tumor core (1 or 3), enhancing tumor (3).
    """
    whole_tumor = (target > 0).float()
    tumor_core = ((target == 1) | (target == 3)).float()
    enhancing_tumor = (target == 3).float()
    return torch.stack([whole_tumor, tumor_core, enhancing_tumor], dim=1)


class DiceLoss(nn.Module):
    def __init__(self, batch_dice: bool = False, region_based: bool = False):
        super().__init__()
        self.batch_dice = batch_dice
        self.region_based = region_based

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.region_based:
            target_onehot = labels_to_regions(target)
        else:
            num_classes = pred.shape[1]
            target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes)
            target_onehot = target_onehot.movedim(-1, 1).float()

        spatial_dims = tuple(range(2, pred.dim()))
        sum_dims = (0,) + spatial_dims if self.batch_dice else spatial_dims

        intersection = (pred * target_onehot).sum(dim=sum_dims)
        denom = pred.sum(dim=sum_dims) + target_onehot.sum(dim=sum_dims)
        dice_per_class = (2.0 * intersection + EPS) / (denom + EPS)

        return 1.0 - dice_per_class.mean()


class DiceCELoss(nn.Module):
    def __init__(self, batch_dice: bool = False):
        super().__init__()
        self.dice = DiceLoss(batch_dice=batch_dice, region_based=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        pred_for_gather = pred.clamp(min=EPS)
        log_pred = torch.log(pred_for_gather)
        ce_loss = torch.nn.functional.nll_loss(log_pred, target)
        return dice_loss + ce_loss


class DiceBCELoss(nn.Module):
    def __init__(self, batch_dice: bool = False):
        super().__init__()
        self.dice = DiceLoss(batch_dice=batch_dice, region_based=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        target_regions = labels_to_regions(target)
        pred_clamped = pred.clamp(min=EPS, max=1.0 - EPS)
        bce_loss = torch.nn.functional.binary_cross_entropy(pred_clamped, target_regions)
        return dice_loss + bce_loss


class DeepSupervisionWrapper(nn.Module):
    """Combines a base loss across the model's multiple deep-supervision
    output resolutions. Weights decay geometrically the further a
    resolution is from full, normalized to sum to 1."""

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, preds: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        n = len(preds)
        raw_weights = [1.0 / (2 ** (n - 1 - i)) for i in range(n)]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

        total_loss = 0.0
        for pred, weight in zip(preds, weights):
            if pred.shape[2:] != target.shape[1:]:
                target_ds = torch.nn.functional.interpolate(
                    target.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest"
                ).squeeze(1).long()
            else:
                target_ds = target
            total_loss = total_loss + weight * self.base_loss(pred, target_ds)
        return total_loss


def build_loss(training_cfg, model_cfg) -> nn.Module:
    loss_type = getattr(training_cfg, "loss_type", "dice_ce")
    batch_dice = getattr(model_cfg, "batch_dice", False)

    if loss_type == "dice_ce":
        return DiceCELoss(batch_dice=batch_dice)
    if loss_type == "dice_bce":
        return DiceBCELoss(batch_dice=batch_dice)
    raise ValueError(f"unknown loss_type: {loss_type}")


if __name__ == "__main__":
    from src.config_schema.model.model_schema import ModelConfig
    from src.config_schema.training.training_schema import TrainingConfig

    loss_fn = build_loss(TrainingConfig(), ModelConfig())
    pred = torch.softmax(torch.randn(2, 3, 16, 16, 16), dim=1)
    target = torch.randint(0, 3, (2, 16, 16, 16))
    print("real dice_ce loss value:", loss_fn(pred, target).item())

    from src.config_schema.model.model_schema import Nnunet3DModelSchema
    from src.config_schema.training.training_schema import NnunetBaselineTrainingSchema

    region_training_cfg = NnunetBaselineTrainingSchema(loss_type="dice_bce")
    region_model_cfg = Nnunet3DModelSchema(region_based_training=True)
    region_loss_fn = build_loss(region_training_cfg, region_model_cfg)
    region_pred = torch.sigmoid(torch.randn(2, 3, 16, 16, 16))
    print("real dice_bce loss value:", region_loss_fn(region_pred, target).item())