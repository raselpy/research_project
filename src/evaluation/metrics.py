"""Segmentation evaluation metrics."""
import numpy as np
from medpy.metric.binary import hd95 as _medpy_hd95


def hausdorff95(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_spacing) -> float:
    """95th-percentile Hausdorff distance in mm.

    Uses medpy.metric.binary.hd95, which computes the true 95th percentile
    of the symmetric surface-distance distribution — NOT max(directed
    Hausdorff), which is a different (and wrong, for BraTS comparison)
    metric a hand-rolled version could easily compute by mistake.

    Returns 0.0 if pred and gt are both empty (perfect match), and BraTS's
    own sentinel value (373.13) — not inf — if exactly one of pred/gt is
    empty, matching the BraTS evaluation platform's own convention.
    """
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return 373.13
    return float(_medpy_hd95(pred_mask, gt_mask, voxelspacing=voxel_spacing))


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice similarity coefficient. Returns 1.0 if both masks are empty."""
    if not pred_mask.any() and not gt_mask.any():
        return 1.0
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)