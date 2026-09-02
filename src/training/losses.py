"""Loss functions for training. Section 2.3's Dice+CE (standard) and
Dice+BCE (region-based training ablation), Section 2.4's batch-Dice vs.
sample-Dice (BD ablation), and a deep-supervision wrapper that combines
losses across the model's multiple output resolutions.

IMPORTANT — probabilities, not logits: NNUNet3D's SegmentationHead
already applies softmax/sigmoid internally (Fig. 1's "1x1x1 conv -
softmax" block), so `pred` here is always a probability tensor (softmax
output summing to 1 across the class dim, or independent per-region
sigmoid outputs in [0,1]) — never raw logits. This is why the loss
functions use NLL-style / plain BCE math rather than
nn.CrossEntropyLoss/nn.BCEWithLogitsLoss, which both expect logits and
apply their own internal nonlinearity.
"""

import torch
from torch import nn

EPS = 1e-5


def labels_to_regions(target: torch.Tensor) -> torch.Tensor:
    """Converts contiguous class-index labels {0,1,2,3} (background, NCR,
    ED, ET — see src/datasets/prepare.py's remap_labels) into the 3
    partially-overlapping BraTS evaluation regions (Section 2.3):
    whole tumor (any of 1,2,3), tumor core (1 or 3), enhancing tumor (3).

    target: (N, D, H, W) long tensor of class indices.
    returns: (N, 3, D, H, W) float tensor, channel order [WT, TC, ET].
    """
    whole_tumor = (target > 0).float()
    tumor_core = ((target == 1) | (target == 3)).float()
    enhancing_tumor = (target == 3).float()
    return torch.stack([whole_tumor, tumor_core, enhancing_tumor], dim=1)


class DiceLoss(nn.Module):
    """Soft Dice loss on probability-valued predictions.

    batch_dice=False (default, "sample Dice"): computes Dice per sample
    in the batch independently, then averages — small-foreground samples
    can dominate gradients (Section 2.4).
    batch_dice=True (the BD ablation): treats the whole batch as one
    large sample before computing Dice, regularizing against that.
    """

    def __init__(self, batch_dice: bool = False, region_based: bool = False):
        super().__init__()
        self.batch_dice = batch_dice
        self.region_based = region_based

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.region_based:
            # pred: (N, 3, ...) independent sigmoid probabilities per region.
            # target: (N, D, H, W) class indices -> converted to (N, 3, ...) here.
            target_onehot = labels_to_regions(target)
        else:
            # pred: (N, C, ...) softmax probabilities over mutually exclusive classes.
            num_classes = pred.shape[1]
            target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes)
            target_onehot = target_onehot.movedim(-1, 1).float()  # (N, C, D, H, W)

        spatial_dims = tuple(range(2, pred.dim()))
        if self.batch_dice:
            sum_dims = (0,) + spatial_dims  # collapse batch + spatial, keep class dim
        else:
            sum_dims = spatial_dims  # keep batch + class dims, average over batch after

        intersection = (pred * target_onehot).sum(dim=sum_dims)
        denom = pred.sum(dim=sum_dims) + target_onehot.sum(dim=sum_dims)
        dice_per_class = (2.0 * intersection + EPS) / (denom + EPS)

        return 1.0 - dice_per_class.mean()


class DiceCELoss(nn.Module):
    """Standard formulation: Dice + cross-entropy, both operating on the
    3 contiguous class labels (background/NCR/ED/ET)."""

    def __init__(self, batch_dice: bool = False):
        super().__init__()
        self.dice = DiceLoss(batch_dice=batch_dice, region_based=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        # NLL on already-softmaxed probabilities (equivalent to cross-entropy
        # given pred is already a valid probability distribution over classes).
        pred_for_gather = pred.clamp(min=EPS)
        log_pred = torch.log(pred_for_gather)
        ce_loss = torch.nn.functional.nll_loss(log_pred, target)
        return dice_loss + ce_loss


class DiceBCELoss(nn.Module):
    """Region-based training ablation (Section 2.3): Dice + BCE, both
    operating on the 3 partially-overlapping regions (WT/TC/ET) rather
    than the 3 mutually exclusive classes."""

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
    """Wraps a base loss (DiceCELoss/DiceBCELoss) to combine it across the
    model's multiple deep-supervision output resolutions. Predictions are
    assumed ordered lowest-to-highest resolution (matching NNUNet3D's
    forward() output), with the last element at full input resolution.
    Weights decay geometrically the further a resolution is from full
    (a common deep-supervision weighting), normalized to sum to 1.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, preds: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        n = len(preds)
        raw_weights = [1.0 / (2 ** (n - 1 - i)) for i in range(n)]  # last (full-res) gets weight 1
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

        total_loss: torch.Tensor = torch.zeros((), device=preds[0].device)
        for pred, weight in zip(preds, weights):
            if pred.shape[2:] != target.shape[1:]:
                # downsample target (nearest-neighbor — it's integer labels) to match this resolution
                target_ds = (
                    torch.nn.functional.interpolate(target.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest")
                    .squeeze(1)
                    .long()
                )
            else:
                target_ds = target
            total_loss = total_loss + weight * self.base_loss(pred, target_ds)
        return total_loss


def build_loss(training_cfg, model_cfg) -> nn.Module:
    """Reads TrainingConfig.loss_type ('dice_ce' | 'dice_bce') and
    ModelConfig.batch_dice to build the right base loss. Uses getattr
    with the schema's own defaults so this also works against the bare
    base ModelConfig/TrainingConfig (no paper-specific fields set) —
    not just the full Nnunet3DModelSchema/NnunetBaselineTrainingSchema.
    """
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

    # NOTE: unlike the original plan's snippet (raw torch.randn as `pred`),
    # this uses a softmax-normalized tensor — NNUNet3D's forward() already
    # applies softmax/sigmoid internally (see this module's docstring), so
    # raw unconstrained values are not valid input to a probability-based
    # Dice loss and would produce a meaningless/non-finite result.
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
