"""Real data augmentation presets (Section 2.4), replacing the flip-only
stub. Three presets, selected by TrainingConfig.augmentation_preset:
'baseline' (nnU-Net's own default augmentation — not "no augmentation"),
'DA' (more aggressive, Section 2.4), and 'DA_star' (DA with per-channel
brightness instead of per-sample).
"""

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform


def build_augmentation(preset: str, patch_size=None) -> Compose:
    """preset: 'baseline' (nnU-Net default) | 'DA' | 'DA_star'.

    `patch_size` should be set from cfg.model.patch_size at the real
    training call site (SpatialTransform uses it for the post-augmentation
    random crop); left as None here so the acceptance check below can
    call this standalone without a full training config — batchgenerators
    falls back to the input array's own spatial shape in that case.
    """
    if preset == "baseline":
        rotation_scaling_p, scale_range, elastic_p, brightness_p = 0.2, (0.85, 1.25), 0.0, 0.0
    elif preset in ("DA", "DA_star"):
        rotation_scaling_p, scale_range, elastic_p, brightness_p = 0.3, (0.65, 1.6), 0.3, 0.3
    else:
        raise ValueError(f"unknown augmentation_preset: {preset}")

    transforms = [
        SpatialTransform(
            patch_size=patch_size,
            p_rot_per_sample=rotation_scaling_p,
            p_scale_per_sample=rotation_scaling_p,
            scale=scale_range,
            independent_scale_for_each_axis=True,
            do_elastic_deform=elastic_p > 0,
            p_el_per_sample=elastic_p,
        ),
    ]
    if brightness_p > 0:
        per_channel = preset == "DA_star"
        transforms.append(
            BrightnessTransform(
                mu=0.0,
                sigma=0.1,
                per_channel=per_channel,
                p_per_sample=brightness_p if not per_channel else 1.0,
                p_per_channel=0.5 if per_channel else 1.0,
            )
        )
        transforms.append(GammaTransform(gamma_range=(0.6, 1.6), p_per_sample=0.3))

    return Compose(transforms)


if __name__ == "__main__":
    import numpy as np

    x = {
        "data": np.random.randn(1, 4, 32, 32, 32).astype("float32"),
        "seg": np.random.randint(0, 3, (1, 1, 32, 32, 32)).astype("float32"),
    }

    for preset in ["baseline", "DA", "DA_star"]:
        tf = build_augmentation(preset)
        out = tf(**{k: v.copy() for k, v in x.items()})
        changed = not np.allclose(out["data"], x["data"])
        print(preset, "data changed:", changed)
