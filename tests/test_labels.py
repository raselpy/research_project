import numpy as np

from src.datasets.prepare import remap_labels


def test_float_drift_from_nifti_scaling_does_not_drop_enhancing_tumor():
    """Regression test for the first Phase 3 bug: NIfTI scl_slope/
    scl_inter scaling can turn an exact integer label like 3 into
    2.9999998. An exact-equality check silently dropped every
    enhancing-tumor voxel to background under this condition, confirmed
    against real BraTS-2024-GLI data."""
    seg = np.array([[[0.0, 1.0, 2.0, 2.9999998]]])
    result = remap_labels(seg)
    assert result.flatten().tolist() == [0, 1, 2, 3]


def test_resection_cavity_coexisting_with_enhancing_tumor():
    """Regression test for the second Phase 3 bug: BraTS 2023+/2024 GLI's
    raw label 4 (resection cavity) can be present in the SAME case as
    raw label 3 (enhancing tumor) — confirmed on real data. Enhancing
    tumor must still map to contiguous class 3; resection cavity must
    map to background (0), not be lumped in with 3 as 'unexpected'."""
    seg = np.array([[[0, 1, 2, 3, 4]]])
    result = remap_labels(seg)
    assert result.flatten().tolist() == [0, 1, 2, 3, 0]


def test_genuinely_unexpected_label_still_maps_to_background():
    seg = np.array([[[0, 1, 2, 3, 7]]])
    result = remap_labels(seg)
    assert result.flatten().tolist() == [0, 1, 2, 3, 0]


def test_output_dtype_is_uint8():
    seg = np.array([[[0, 1, 2, 3]]])
    result = remap_labels(seg)
    assert result.dtype == np.uint8
