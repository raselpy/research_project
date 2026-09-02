import numpy as np

from src.evaluation.metrics import dice_score, hausdorff95


def test_hd95_on_one_voxel_shift_is_small_not_inflated():
    """Regression test for the Phase 3 metrics bug: a hand-rolled
    max-Hausdorff implementation would report a large, misleading value
    here. The true 95th-percentile HD95 (medpy) must stay small for a
    tiny, localized perturbation."""
    a = np.zeros((20, 20, 20), dtype=bool)
    a[5:15, 5:15, 5:15] = True
    b = np.zeros((20, 20, 20), dtype=bool)
    b[5:15, 5:15, 5:16] = True
    d = hausdorff95(a, b, voxel_spacing=(1.0, 1.0, 1.0))
    assert 0.0 < d < 5.0


def test_hd95_both_empty_is_zero():
    empty = np.zeros((5, 5, 5), dtype=bool)
    assert hausdorff95(empty, empty, voxel_spacing=(1, 1, 1)) == 0.0


def test_hd95_one_empty_matches_brats_sentinel():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[2:5, 2:5, 2:5] = True
    empty = np.zeros((10, 10, 10), dtype=bool)
    assert hausdorff95(a, empty, voxel_spacing=(1, 1, 1)) == 373.13


def test_dice_identical_masks_is_one():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[2:5, 2:5, 2:5] = True
    assert dice_score(a, a) == 1.0


def test_dice_both_empty_is_one():
    empty = np.zeros((5, 5, 5), dtype=bool)
    assert dice_score(empty, empty) == 1.0
