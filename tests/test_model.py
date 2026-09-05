import torch

from src.config_schema.model.model_schema import Nnunet3DModelSchema
from src.datasets.prepare import RAW_TO_CONTIGUOUS
from src.models.nnunet3d import NNUNet3D


def test_forward_output_shape_matches_input():
    """Minimum bar from the plan's own acceptance check: output spatial
    shape matches input. Uses reduced channels so this runs fast/light
    in CI, not the full paper-scale model."""
    model = NNUNet3D(patch_size=(32, 32, 32), base_num_features=2, max_num_features=16, num_downsampling=3)
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        outs = model(x)
    assert outs[-1].shape[2:] == x.shape[2:]


def test_deep_supervision_head_count_and_shapes():
    """Regression test for the Phase 1 head-indexing bug: heads must
    exist at exactly the resolutions Fig. 1 specifies (all but the two
    lowest decoder resolutions), not an off-by-one set of channels."""
    model = NNUNet3D(patch_size=(128, 128, 128), base_num_features=2, max_num_features=16, num_downsampling=5)
    x = torch.randn(1, 4, 128, 128, 128)
    with torch.no_grad():
        outs = model(x)
    assert len(outs) == 3
    assert [tuple(o.shape[2:]) for o in outs] == [(32, 32, 32), (64, 64, 64), (128, 128, 128)]


def test_full_scale_param_count_matches_paper():
    """Paper states 31.2M params for the full-scale config."""
    model = NNUNet3D()  # paper defaults: patch 128^3, base 32, max 320, 5 downsamples
    n_params = sum(p.numel() for p in model.parameters())
    assert 31_100_000 < n_params < 31_300_000


def test_bottleneck_collapse_guard_raises():
    import pytest

    with pytest.raises(ValueError):
        NNUNet3D(patch_size=(32, 32, 32), num_downsampling=5)  # collapses to bottleneck size 1


def test_empty_deep_supervision_heads_guard_raises():
    """Regression test for the Phase 8 bug: num_downsampling < 3 produces
    zero deep-supervision heads (heads exist for decoder indices
    range(2, num_downsampling), empty below 3). Without this guard, the
    real failure only surfaced four call-frames deep inside
    DeepSupervisionWrapper as a confusing IndexError on an empty preds
    list, discovered via an actual Phase 8 CV smoke-test run."""
    import pytest

    with pytest.raises(ValueError):
        NNUNet3D(num_downsampling=2)


def test_num_classes_matches_prepare_py_label_scheme():
    """Regression test for the Phase 6 bug: Nnunet3DModelSchema's default
    num_classes silently drifted out of sync with the actual number of
    contiguous classes src/datasets/prepare.py's remap_labels() produces
    (background + NCR + ED + ET = 4), causing a RuntimeError only at
    real-data training time, not caught by any structural test until
    then. This test fails immediately if that happens again."""
    max_contiguous_label = max(RAW_TO_CONTIGUOUS.values())  # 3 (ET) — RESECTION_CAVITY_RAW maps to 0, not a new max
    expected_num_classes = max_contiguous_label + 1  # +1 for background/zero-indexing

    default_num_classes = Nnunet3DModelSchema().num_classes
    assert default_num_classes == expected_num_classes, (
        f"Nnunet3DModelSchema.num_classes={default_num_classes} but prepare.py's "
        f"remap_labels() produces labels 0..{max_contiguous_label} "
        f"({expected_num_classes} classes) — these must stay in sync."
    )
