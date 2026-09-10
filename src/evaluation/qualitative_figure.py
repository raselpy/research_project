"""Qualitative results figure using the standard tri-planar + 3D layout
(ITK-SNAP's default 4-pane view): one row each for the best,
75th-percentile, median, 25th-percentile, and worst case (ranked by
whole-tumor Dice), four panels per row — axial, coronal, sagittal (all
three centered on the same synced 3D cursor, with the predicted
segmentation overlaid on T1c), plus a 3D isosurface render. Overlay
colors: edema=violet, enhancing tumor=yellow, necrosis/non-enhancing
tumor=turquoise.

Uses one variant's ensembled holdout predictions (Table 2's pool —
20 genuine holdout cases, matching the paper's own use of its
"Validation set" for this figure, not the training-set CV folds).

Predictions aren't saved anywhere from Phase 9's evaluation run (only
scalar Dice/HD95 were written to eval_results_*.json) — this
re-runs ensemble inference to get the actual predicted volume for
each of the 5 selected cases. Uses a CENTER crop rather than the
random crop evaluation used — a fixed, reproducible, and more
visually sensible choice for a figure than an arbitrary random one;
the Dice values shown in the figure titles are the real evaluated
scores from eval_results_<variant>.json, not re-derived from this
crop, so they stay consistent with Table 2.

NOTE on orientation: image.npy/seg.npy are stored as (channel, H, W, D)
without their original NIfTI affine, so which array axis is truly
anatomical superior-inferior/left-right/anterior-posterior can't be
verified from this script alone. Axial is displayed along the D axis
(matching the slice this script always used); coronal/sagittal use the
remaining two axes. If the panels look anatomically transposed against
a known-good viewer (ITK-SNAP, Slicer) for your data, the fix is a
one-line axis relabel here, not a re-derivation of anything upstream.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage import measure

from src.config_schema import setup_config
from src.evaluation.ensemble import ensemble_predict
from src.evaluation.run import pred_to_regions

setup_config()

# edema=violet, enhancing tumor=yellow, necrosis/non-enhancing tumor=turquoise.
CLASS_COLORS = {
    0: (0, 0, 0, 0),  # background: fully transparent
    1: (0.25, 0.88, 0.82, 0.6),  # NCR -> turquoise
    2: (0.56, 0.0, 1.0, 0.5),  # ED -> violet
    3: (1.0, 0.95, 0.0, 0.7),  # ET -> yellow
}


def _center_crop(volume: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    """Deterministic center crop from a (C, H, W, D) volume — chosen
    over a random crop for a figure because it's reproducible and
    consistently centers on the brain, not because it matches the
    random crop used during evaluation (the Dice values shown in the
    figure come from eval_results_*.json, not re-derived here)."""
    spatial_shape = volume.shape[1:]
    starts = [max((dim - p) // 2, 0) for dim, p in zip(spatial_shape, patch_size)]
    slices = tuple(slice(s, s + p) for s, p in zip(starts, patch_size))
    cropped = volume[(slice(None),) + slices]
    pad_widths = [(0, 0)] + [(0, max(p - c, 0)) for p, c in zip(patch_size, cropped.shape[1:])]
    if any(w[1] > 0 for w in pad_widths):
        cropped = np.pad(cropped, pad_widths, mode="constant")
    return cropped


def regions_to_class_map(pred_regions: dict[str, np.ndarray]) -> np.ndarray:
    """Converts WT/TC/ET region masks (region-based sigmoid output) back
    to a single per-voxel class map {0,1,2,3} for coloring — the same
    class scheme prepare.py writes: 0=background, 1=NCR, 2=ED, 3=ET.
    ET takes priority, then TC-minus-ET is NCR, then WT-minus-TC is ED."""
    whole, core, enhancing = pred_regions["whole"], pred_regions["core"], pred_regions["enhancing"]
    class_map = np.zeros_like(whole, dtype=np.uint8)
    class_map[whole & ~core] = 2  # ED
    class_map[core & ~enhancing] = 1  # NCR
    class_map[enhancing] = 3  # ET
    return class_map


def _shade_faces(verts: np.ndarray, faces: np.ndarray, base_rgba: tuple, light_dir: np.ndarray) -> np.ndarray:
    """Per-face Lambertian shading. matplotlib's Poly3DCollection paints
    every face the same flat color by default, which is the main reason
    a marching-cubes mesh reads as a flat colored blob instead of a solid
    3D shape — there's no lighting cue for curvature. This computes each
    triangle's normal, dots it against a fixed light direction, and
    darkens/brightens that face's copy of the base color accordingly."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    intensity = np.clip(normals @ light_dir, 0.0, 1.0)
    intensity = 0.35 + 0.65 * intensity  # ambient floor so faces facing away don't go pure black
    r, g, b, a = base_rgba
    return np.column_stack([r * intensity, g * intensity, b * intensity, np.full_like(intensity, a)])


def plot_3d_segmentation(
    ax, class_map: np.ndarray, colors: dict, downsample: int = 1, smooth_sigma: float = 1.0
) -> None:
    """Renders each tumor class as a shaded 3D isosurface (marching
    cubes on the per-class binary mask) in a shared 3D axes, using the
    same class -> color legend as the 2D overlay panel.

    Two things matter here beyond just calling marching_cubes:

    - Raw binary masks from a noisy/fragmented prediction (e.g. a
      no-data-aug ablation) produce dozens of tiny disconnected voxel
      islands, which marching_cubes turns into a "cloud of dust" that's
      unreadable as a 3D shape. Gaussian-smoothing the mask before
      thresholding merges nearby fragments into continuous surfaces —
      this changes only how the shape is drawn, not the underlying
      prediction or any reported Dice/HD95 number, which come from
      eval_results_*.json.
    - Flat, unshaded, unlit faces (matplotlib's Poly3DCollection default)
      give no depth cue at all — a sphere and a flat disc look identical
      head-on. `_shade_faces` fixes that with simple per-face lighting.

    ED (edema) is drawn as a translucent outer shell so the nested
    NCR/ET regions are visible inside it; NCR and ET are drawn solid.
    matplotlib's 3D backend doesn't do true depth-buffered transparency
    (it depth-sorts whole collections, not per-triangle), so stacking
    more than one translucent nested surface produces muddy/incorrect
    occlusion — solid inner regions + a single translucent outer shell,
    drawn last, is what actually reads correctly.

    `downsample` strides the volume before marching cubes purely for
    render speed/file size on very large volumes; it does not affect
    any reported metric."""
    light_dir = np.array([0.4, -0.5, 0.75])
    light_dir = light_dir / np.linalg.norm(light_dir)
    vol = class_map[::downsample, ::downsample, ::downsample]
    drew_any = False
    for cls in (1, 3, 2):  # NCR, ET (solid) first, then ED (translucent shell) on top
        mask = (vol == cls).astype(np.float32)
        # marching_cubes needs a real surface to extract; skip near-empty masks
        # (common for enhancing tumor in small/worst-case predictions) rather
        # than letting it raise.
        if mask.sum() < 15:
            continue
        smoothed = ndimage.gaussian_filter(mask, sigma=smooth_sigma)
        if smoothed.max() < 0.5:
            continue  # smoothing diluted an already-tiny fragment below the isosurface level
        try:
            verts, faces, _, _ = measure.marching_cubes(smoothed, level=0.5)
        except (ValueError, RuntimeError):
            continue
        r, g, b, _ = colors[cls]
        alpha = 0.35 if cls == 2 else 1.0  # only the outer ED shell is translucent
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor(_shade_faces(verts, faces, (r, g, b, alpha), light_dir))
        mesh.set_edgecolor((0, 0, 0, 0))
        ax.add_collection3d(mesh)
        drew_any = True
    ax.set_xlim(0, vol.shape[0])
    ax.set_ylim(0, vol.shape[1])
    ax.set_zlim(0, vol.shape[2])
    ax.set_box_aspect(vol.shape)
    # Light gray panes + faint gridlines (instead of set_axis_off()) give the
    # three receding walls of a box — the main depth cue that was missing
    # before; without it a render can look like a flat 2D blob regardless of
    # how the mesh itself is lit.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.96, 0.97, 1.0))
        axis.pane.set_edgecolor((0.7, 0.7, 0.7, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-55)
    if not drew_any:
        ax.text2D(0.5, 0.5, "no tumor\npredicted", ha="center", va="center", fontsize=7, transform=ax.transAxes)


def pick_display_coords(seg_class_map: np.ndarray) -> tuple[int, int, int]:
    """3D coordinate of the ground-truth tumor's center of mass, used as
    the shared cursor position for all three orthogonal planes — the
    same synced-cursor convention ITK-SNAP's default 4-pane view uses
    (move the cursor in one plane, the other two jump to match). Falls
    back to the volume's center if there's no tumor at all (shouldn't
    happen for cases selected by whole_dice, but avoids a NaN coordinate)."""
    tumor_voxels = np.argwhere(seg_class_map > 0)
    if len(tumor_voxels) == 0:
        return tuple(s // 2 for s in seg_class_map.shape)
    centroid = tumor_voxels.mean(axis=0)
    return tuple(int(round(c)) for c in centroid)


def select_percentile_cases(eval_results_path: Path) -> list[dict]:
    """Reads eval_results_<variant>.json (the holdout ensemble scores),
    sorts by whole_dice, and picks best/75th-pct/median/25th-pct/worst
    by nearest-rank — same 5-row selection as the paper's Figure 2."""
    with open(eval_results_path) as f:
        cases = json.load(f)
    cases_sorted = sorted(cases, key=lambda c: c["whole_dice"], reverse=True)
    n = len(cases_sorted)
    indices = {
        "Best": 0,
        "75th percentile": round(0.25 * (n - 1)),
        "Median": round(0.50 * (n - 1)),
        "25th percentile": round(0.75 * (n - 1)),
        "Worst": n - 1,
    }
    return [{"label": label, **cases_sorted[idx]} for label, idx in indices.items()]


def build_figure(
    variant: str,
    cfg,
    checkpoint_glob: str,
    processed_dir: Path,
    eval_results_path: Path,
    device: torch.device,
    out_path: Path,
) -> None:
    import glob

    checkpoint_paths = sorted(glob.glob(checkpoint_glob))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched: {checkpoint_glob}")

    selected = select_percentile_cases(eval_results_path)
    patch_size = tuple(cfg.model.patch_size)

    # 4 columns: axial, coronal, sagittal (each T1c + overlay), 3D render.
    # The 3D column needs its own projection="3d" axes, which plt.subplots
    # can't mix in with the other 2D axes in one call, so build the grid
    # by hand.
    fig = plt.figure(figsize=(12, 3 * len(selected)))
    axes = np.empty((len(selected), 3), dtype=object)
    axes_3d = []
    for row in range(len(selected)):
        for col in range(3):
            axes[row, col] = fig.add_subplot(len(selected), 4, row * 4 + col + 1)
        axes_3d.append(fig.add_subplot(len(selected), 4, row * 4 + 4, projection="3d"))
    cmap = ListedColormap([CLASS_COLORS[i] for i in range(4)])
    column_titles = ["Axial", "Coronal", "Sagittal"]

    for row, case in enumerate(selected):
        case_id = case["case_id"]
        case_dir = processed_dir / case_id
        image = np.load(case_dir / "image.npy")  # (4, H, W, D): t1, t1ce, t2, flair
        seg = np.load(case_dir / "seg.npy")  # (H, W, D), ground truth class labels

        image_crop = _center_crop(image, patch_size)
        seg_crop = _center_crop(seg[np.newaxis], patch_size)[0]

        batch = torch.from_numpy(image_crop).float().unsqueeze(0)
        pred = ensemble_predict(checkpoint_paths, cfg, batch, device)
        pred_regions = pred_to_regions(pred, cfg.model.region_based_training)
        pred_class_map = regions_to_class_map(pred_regions)

        t1c = image_crop[1]  # (H, W, D) — base image for all 3 planes, same choice as before
        h_idx, w_idx, d_idx = pick_display_coords(seg_crop)

        # (base_slice, pred_slice) per plane. Coronal/sagittal are transposed
        # so the D axis (this script's "axial" axis) is drawn vertically,
        # matching the convention that the S-I axis runs top-to-bottom in a
        # tri-planar viewer — see the orientation caveat in the module docstring.
        panels = [
            (t1c[:, :, d_idx], pred_class_map[:, :, d_idx]),  # axial: (H, W)
            (t1c[h_idx, :, :].T, pred_class_map[h_idx, :, :].T),  # coronal: (D, W)
            (t1c[:, w_idx, :].T, pred_class_map[:, w_idx, :].T),  # sagittal: (D, H)
        ]

        title = (
            f"{case['label']}: {case_id}, whole {case['whole_dice'] * 100:.2f}, "
            f"core {case['core_dice'] * 100:.2f}, enh {case['enhancing_dice'] * 100:.2f}"
        )
        for col, (base_slice, pred_slice) in enumerate(panels):
            axes[row, col].imshow(base_slice, cmap="gray")
            axes[row, col].imshow(pred_slice, cmap=cmap, vmin=0, vmax=3)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(column_titles[col], fontsize=9)
        axes[row, 0].set_ylabel(case["label"], fontsize=9)
        axes[row, 0].text(0.02, 0.98, title, transform=axes[row, 0].transAxes, fontsize=7, color="yellow", va="top")

        plot_3d_segmentation(axes_3d[row], pred_class_map, CLASS_COLORS)
        if row == 0:
            axes_3d[row].set_title("3D render", fontsize=9)

    # Shared legend for the class colors, since the 3D panel has no
    # colorbar/axis of its own to read colors off of.
    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=CLASS_COLORS[c], label=name)
        for c, name in ((1, "NCR"), (2, "ED"), (3, "ET"))
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"Qualitative results: {variant} (ensembled, holdout set)", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="ablation_dataug_star_bn")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    args, overrides = parser.parse_known_args()

    config_dir = str(Path("configs").resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[f"experiment={args.variant}", *overrides])
    cfg.model.patch_size = list(args.patch_size)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    build_figure(
        variant=args.variant,
        cfg=cfg,
        checkpoint_glob=f"results/checkpoints/{args.variant}/fold_*.pt",
        processed_dir=Path("data/processed"),
        eval_results_path=Path(f"results/tables/eval_results_{args.variant}.json"),
        device=device,
        out_path=Path(f"results/tables/qualitative_{args.variant}.png"),
    )
