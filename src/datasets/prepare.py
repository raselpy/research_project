# """Data preparation: loads raw BraTS-2024-GLI cases, applies the paper's
# exact normalization ("subtract mean, divide by std of nonzero brain
# voxels; non-brain stays 0", Section 2.2), remaps BraTS labels to
# contiguous class indices, and writes processed volumes + a manifest.
#
# Naming convention (BraTS-2024-GLI, confirmed against real data — NOT the
# older BraTS-2020 `_t1.nii.gz` style):
#     <case_id>-t1n.nii   T1 native
#     <case_id>-t1c.nii   T1 contrast-enhanced
#     <case_id>-t2w.nii   T2 weighted
#     <case_id>-t2f.nii   T2 FLAIR
#     <case_id>-seg.nii   segmentation mask
# """
#
# import argparse
# import csv
# from pathlib import Path
# from typing import cast
#
# import nibabel as nib
# import numpy as np
#
# from src.logging_utils.setup import get_logger
#
# logger = get_logger(__name__)
#
# MODALITY_SUFFIXES: dict[str, str] = {
#     "t1": "-t1n.nii",
#     "t1ce": "-t1c.nii",
#     "t2": "-t2w.nii",
#     "flair": "-t2f.nii",
# }
# SEG_SUFFIX = "-seg.nii"
# MODALITY_ORDER: list[str] = [
#     "t1",
#     "t1ce",
#     "t2",
#     "flair",
# ]  # fixed channel order for the model's in_channels=4
#
# # Raw BraTS label values -> contiguous indices for CrossEntropyLoss.
# #   0 = background
# #   1 = NCR (necrotic tumor core)
# #   2 = ED (peritumoral edema)
# #   3 = ET (enhancing tumor)
# #   4 = resection cavity — introduced in BraTS 2023+/2024 GLI for
# #       post-surgical cases; NOT part of the original 2020 paper's
# #       whole-tumor/tumor-core/enhancing-tumor formulation, and not
# #       part of any of the 3 evaluated regions. Confirmed against real
# #       data: many cases genuinely have BOTH 3 and 4 present at once
# #       (post-op cases with a resection cavity *and* residual/recurrent
# #       enhancing tumor) — this is not noise or float drift, it's real.
# #       Mapped to background (0) since it has no place in the paper's
# #       3-region formulation.
# RAW_TO_CONTIGUOUS = {0: 0, 1: 1, 2: 2, 3: 3}
# RESECTION_CAVITY_RAW = 4
#
#
# def normalize_modality(volume: np.ndarray) -> np.ndarray:
#     """Subtract mean, divide by std of nonzero (brain) voxels. Non-brain
#     (zero) voxels remain 0, exactly as Section 2.2 describes."""
#     brain_mask = volume != 0
#     if not brain_mask.any():
#         return volume.astype(np.float32)
#     brain_voxels = volume[brain_mask]
#     mean = brain_voxels.mean()
#     std = brain_voxels.std()
#     normalized = np.zeros_like(volume, dtype=np.float32)
#     if std > 0:
#         normalized[brain_mask] = (brain_voxels - mean) / std
#     else:
#         normalized[brain_mask] = 0.0
#     return normalized
#
#
# def remap_labels(seg: np.ndarray) -> np.ndarray:
#     """Maps raw BraTS label values to contiguous {0,1,2,3} indices.
#
#     Rounds to the nearest integer before comparing (not exact equality):
#     NIfTI's scl_slope/scl_inter scaling, applied when reading via
#     nibabel, can turn an exact integer label like 3 into something like
#     2.9999998 — confirmed against real BraTS-2024-GLI data, where an
#     exact-equality check silently dropped enhancing tumor voxels to
#     background. Resection cavity (raw label 4, present in some
#     post-surgical cases alongside 3) is intentionally mapped to
#     background — see RESECTION_CAVITY_RAW's comment above — not treated
#     as an anomaly.
#     """
#     seg_int = np.rint(seg).astype(np.int64)
#     raw_values = set(np.unique(seg_int).tolist())
#
#     remap = dict(RAW_TO_CONTIGUOUS)
#     if RESECTION_CAVITY_RAW in raw_values:
#         remap[RESECTION_CAVITY_RAW] = 0  # explicit, not silent — see module-level comment
#
#     out = np.zeros_like(seg_int, dtype=np.uint8)
#     for raw_val, contiguous_val in remap.items():
#         out[seg_int == raw_val] = contiguous_val
#
#     truly_unexpected = raw_values - set(remap.keys())
#     if truly_unexpected:
#         logger.warning(
#             f"Truly unexpected raw label values {truly_unexpected} found "
#             f"(not in {{0,1,2,3,4}}); left as background (0)."
#         )
#
#     return out
#
#
# def find_case_dirs(dataset_path: Path) -> list[Path]:
#     return sorted(p for p in dataset_path.iterdir() if p.is_dir())
#
#
# def process_case(case_dir: Path, out_root: Path) -> dict:
#     case_id = case_dir.name
#     modality_volumes = []
#     affine = None
#     for modality in MODALITY_ORDER:
#         modality_path = case_dir / f"{case_id}{MODALITY_SUFFIXES[modality]}"
#         # nib.load()'s return type is the generic FileBasedImage per
#         # nibabel's stubs, which doesn't statically expose
#         # .affine/.get_fdata()/.dataobj even though every real .nii file
#         # loads as Nifti1Image, which does. cast() reflects the actual
#         # runtime type without silencing unrelated type errors elsewhere.
#         img = cast(nib.Nifti1Image, nib.load(str(modality_path)))
#         if affine is None:
#             affine = img.affine
#         volume = img.get_fdata(dtype=np.float32)
#         modality_volumes.append(normalize_modality(volume))
#
#     seg_path = case_dir / f"{case_id}{SEG_SUFFIX}"
#     seg_img = cast(nib.Nifti1Image, nib.load(str(seg_path)))
#     seg = remap_labels(np.asarray(seg_img.dataobj))
#
#     stacked = np.stack(modality_volumes, axis=0)  # (4, H, W, D)
#
#     case_out_dir = out_root / case_id
#     case_out_dir.mkdir(parents=True, exist_ok=True)
#     np.save(case_out_dir / "image.npy", stacked)
#     np.save(case_out_dir / "seg.npy", seg)
#
#     foreground_voxels = int((seg > 0).sum())
#     return {
#         "case_id": case_id,
#         "image_path": str(case_out_dir / "image.npy"),
#         "seg_path": str(case_out_dir / "seg.npy"),
#         "shape": "x".join(str(d) for d in stacked.shape[1:]),
#         "foreground_voxels": foreground_voxels,
#     }
#
#
# def main(dataset_path: str) -> None:
#     dataset_dir = Path(dataset_path)
#     out_root = Path("data/processed")
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     case_dirs = find_case_dirs(dataset_dir)
#     logger.info(f"Found {len(case_dirs)} case(s) under {dataset_dir}")
#
#     manifest_rows = []
#     for case_dir in case_dirs:
#         try:
#             row = process_case(case_dir, out_root)
#             manifest_rows.append(row)
#             logger.info(
#                 f"Processed {row['case_id']}: shape={row['shape']}, foreground_voxels={row['foreground_voxels']}"
#             )
#         except FileNotFoundError as e:
#             logger.warning(f"Skipping {case_dir.name}: missing expected file ({e})")
#
#     manifest_path = out_root / "manifest.csv"
#     with open(manifest_path, "w", newline="") as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=[
#                 "case_id",
#                 "image_path",
#                 "seg_path",
#                 "shape",
#                 "foreground_voxels",
#             ],
#         )
#         writer.writeheader()
#         writer.writerows(manifest_rows)
#
#     logger.info(f"Wrote manifest for {len(manifest_rows)} case(s) to {manifest_path}")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--path",
#         type=str,
#         required=True,
#         help="Path to the raw dataset directory (contains case subfolders)",
#     )
#     args = parser.parse_args()
#     main(args.path)

"""Data preparation: loads raw BraTS-2024-GLI cases, applies the paper's
exact normalization ("subtract mean, divide by std of nonzero brain
voxels; non-brain stays 0", Section 2.2), remaps BraTS labels to
contiguous class indices, and writes processed volumes + a manifest.

Naming convention (BraTS-2024-GLI, confirmed against real data — NOT the
older BraTS-2020 `_t1.nii.gz` style):
    <case_id>-t1n.nii   T1 native
    <case_id>-t1c.nii   T1 contrast-enhanced
    <case_id>-t2w.nii   T2 weighted
    <case_id>-t2f.nii   T2 FLAIR
    <case_id>-seg.nii   segmentation mask
"""

import argparse
import csv
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold

from src.logging_utils.setup import get_logger

logger = get_logger(__name__)

MODALITY_SUFFIXES: dict[str, str] = {
    "t1": "-t1n.nii",
    "t1ce": "-t1c.nii",
    "t2": "-t2w.nii",
    "flair": "-t2f.nii",
}
SEG_SUFFIX = "-seg.nii"
MODALITY_ORDER: list[str] = [
    "t1",
    "t1ce",
    "t2",
    "flair",
]  # fixed channel order for the model's in_channels=4

# Raw BraTS label values -> contiguous indices for CrossEntropyLoss.
#   0 = background
#   1 = NCR (necrotic tumor core)
#   2 = ED (peritumoral edema)
#   3 = ET (enhancing tumor)
#   4 = resection cavity — introduced in BraTS 2023+/2024 GLI for
#       post-surgical cases; NOT part of the original 2020 paper's
#       whole-tumor/tumor-core/enhancing-tumor formulation, and not
#       part of any of the 3 evaluated regions. Confirmed against real
#       data: many cases genuinely have BOTH 3 and 4 present at once
#       (post-op cases with a resection cavity *and* residual/recurrent
#       enhancing tumor) — this is not noise or float drift, it's real.
#       Mapped to background (0) since it has no place in the paper's
#       3-region formulation.
RAW_TO_CONTIGUOUS = {0: 0, 1: 1, 2: 2, 3: 3}
RESECTION_CAVITY_RAW = 4


def normalize_modality(volume: np.ndarray) -> np.ndarray:
    """Subtract mean, divide by std of nonzero (brain) voxels. Non-brain
    (zero) voxels remain 0, exactly as Section 2.2 describes."""
    brain_mask = volume != 0
    if not brain_mask.any():
        return volume.astype(np.float32)
    brain_voxels = volume[brain_mask]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    normalized = np.zeros_like(volume, dtype=np.float32)
    if std > 0:
        normalized[brain_mask] = (brain_voxels - mean) / std
    else:
        normalized[brain_mask] = 0.0
    return normalized


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Maps raw BraTS label values to contiguous {0,1,2,3} indices.

    Rounds to the nearest integer before comparing (not exact equality):
    NIfTI's scl_slope/scl_inter scaling, applied when reading via
    nibabel, can turn an exact integer label like 3 into something like
    2.9999998 — confirmed against real BraTS-2024-GLI data, where an
    exact-equality check silently dropped enhancing tumor voxels to
    background. Resection cavity (raw label 4, present in some
    post-surgical cases alongside 3) is intentionally mapped to
    background — see RESECTION_CAVITY_RAW's comment above — not treated
    as an anomaly.
    """
    seg_int = np.rint(seg).astype(np.int64)
    raw_values = set(np.unique(seg_int).tolist())

    remap = dict(RAW_TO_CONTIGUOUS)
    if RESECTION_CAVITY_RAW in raw_values:
        remap[RESECTION_CAVITY_RAW] = 0  # explicit, not silent — see module-level comment

    out = np.zeros_like(seg_int, dtype=np.uint8)
    for raw_val, contiguous_val in remap.items():
        out[seg_int == raw_val] = contiguous_val

    truly_unexpected = raw_values - set(remap.keys())
    if truly_unexpected:
        logger.warning(
            f"Truly unexpected raw label values {truly_unexpected} found "
            f"(not in {{0,1,2,3,4}}); left as background (0)."
        )

    return out


def find_case_dirs(dataset_path: Path) -> list[Path]:
    return sorted(p for p in dataset_path.iterdir() if p.is_dir())


def process_case(case_dir: Path, out_root: Path) -> dict:
    case_id = case_dir.name
    modality_volumes = []
    affine = None
    for modality in MODALITY_ORDER:
        modality_path = case_dir / f"{case_id}{MODALITY_SUFFIXES[modality]}"
        # nib.load()'s return type is the generic FileBasedImage per
        # nibabel's stubs, which doesn't statically expose
        # .affine/.get_fdata()/.dataobj even though every real .nii file
        # loads as Nifti1Image, which does. cast() reflects the actual
        # runtime type without silencing unrelated type errors elsewhere.
        img = cast(nib.Nifti1Image, nib.load(str(modality_path)))
        if affine is None:
            affine = img.affine
        volume = img.get_fdata(dtype=np.float32)
        modality_volumes.append(normalize_modality(volume))

    seg_path = case_dir / f"{case_id}{SEG_SUFFIX}"
    seg_img = cast(nib.Nifti1Image, nib.load(str(seg_path)))
    seg = remap_labels(np.asarray(seg_img.dataobj))

    stacked = np.stack(modality_volumes, axis=0)  # (4, H, W, D)

    case_out_dir = out_root / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_out_dir / "image.npy", stacked)
    np.save(case_out_dir / "seg.npy", seg)

    foreground_voxels = int((seg > 0).sum())
    return {
        "case_id": case_id,
        "image_path": str(case_out_dir / "image.npy"),
        "seg_path": str(case_out_dir / "seg.npy"),
        "shape": "x".join(str(d) for d in stacked.shape[1:]),
        "foreground_voxels": foreground_voxels,
    }


def assign_folds(case_ids: list[str], num_folds: int, fold_seed: int) -> dict[str, int]:
    """Deterministic k-fold assignment over the given case IDs via
    sklearn's KFold(shuffle=True, random_state=fold_seed). Computed once
    here and written into the manifest's `fold` column, rather than
    recomputed per training run — recomputing it per-run risks a
    seed/library-version drift silently producing 5 non-partitioning
    splits (Phase 8, 8.1)."""
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=fold_seed)
    fold_of_case: dict[str, int] = {}
    indices = np.arange(len(case_ids))
    for fold_idx, (_train_idx, val_idx) in enumerate(kf.split(indices)):
        for i in val_idx:
            fold_of_case[case_ids[i]] = fold_idx
    return fold_of_case


def main(dataset_path: str, num_folds: int = 5, fold_seed: int = 42) -> None:
    dataset_dir = Path(dataset_path)
    out_root = Path("data/processed")
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = find_case_dirs(dataset_dir)
    logger.info(f"Found {len(case_dirs)} case(s) under {dataset_dir}")

    manifest_rows = []
    for case_dir in case_dirs:
        try:
            row = process_case(case_dir, out_root)
            manifest_rows.append(row)
            logger.info(
                f"Processed {row['case_id']}: shape={row['shape']}, foreground_voxels={row['foreground_voxels']}"
            )
        except FileNotFoundError as e:
            logger.warning(f"Skipping {case_dir.name}: missing expected file ({e})")

    fold_of_case = assign_folds([row["case_id"] for row in manifest_rows], num_folds, fold_seed)
    for row in manifest_rows:
        row["fold"] = fold_of_case[row["case_id"]]

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "image_path",
                "seg_path",
                "shape",
                "foreground_voxels",
                "fold",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(
        f"Wrote manifest for {len(manifest_rows)} case(s) to {manifest_path} "
        f"({num_folds}-fold assignment, seed={fold_seed})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the raw dataset directory (contains case subfolders)",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    args = parser.parse_args()
    main(args.path, num_folds=args.num_folds, fold_seed=args.fold_seed)
