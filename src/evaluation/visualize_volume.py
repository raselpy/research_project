# """Standalone viewer for a single case's MRI volume — raw .nii/.nii.gz
# (loaded with nibabel) or this project's processed .npy arrays.
#
# This is a debugging/inspection tool, deliberately separate from
# qualitative_figure.py: that script re-runs ensemble inference to render
# *model predictions* for the 5 percentile-selected holdout cases, whereas
# this one just looks at the data itself (any modality, any case, with or
# without the ground-truth segmentation) and needs no checkpoints, no GPU,
# and no eval_results_*.json.
#
# Two display modes, both following the standard convention of a
# color-coded semi-transparent overlay on a grayscale base image:
#
#   --mode triplanar  (default) axial/coronal/sagittal through one shared
#                     cursor — the ITK-SNAP 4-pane layout, plus a 3D
#                     isosurface render when a segmentation is available.
#   --mode grid       a montage of evenly-spaced axial slices, for
#                     scanning a whole volume at once.
#
# Raw vs processed inputs differ in ways that matter here:
#
#   raw .nii       original intensities, original (H, W, D) shape, raw
#                  BraTS label values {0,1,2,4}, and a real affine.
#   processed .npy image.npy is (4, H, W, D) z-scored per modality in the
#                  fixed order t1, t1ce, t2, flair; seg.npy holds labels
#                  already remapped from raw BraTS-2024-GLI values (see
#                  below) to contiguous {0,1,2,3}; the affine is NOT
#                  saved, so orientation can't be recovered from it.
#
# This dataset (BraTS-2024-GLI) already labels segmentations {0,1,2,3}
# directly (NCR=1, ED=2, ET=3) — NOT the {1,2,4} scheme older BraTS
# (2017-2021) used. Raw label 4 here means resection cavity, present in
# some post-surgical cases, and is treated as background, matching
# src/datasets/prepare.py's own remap_labels(). --raw-labels applies
# that same handling to a .npy segmentation; it's automatic for .nii
# input, since a .nii segmentation is always in raw label space.
#
# Whenever a segmentation is given, this also writes a companion
# <name>_3d.html next to the PNG: a rotatable/zoomable 3D mesh (Plotly,
# loaded from a CDN — needs internet the first time it's opened, nothing
# else, no pip install). This exists because a static render can only
# ever approximate depth with one fixed camera angle and fixed lighting —
# real anatomy (edema is often a thin curved sheet, not a round blob) can
# look flat from that one angle no matter how it's lit. Letting you
# actually drag to rotate it removes that ambiguity.
#
# Examples:
#     python -m src.evaluation.visualize_volume \\
#         --image data/raw/BraTS-GLI-02062-102/BraTS-GLI-02062-102-t2w.nii
#
#     python -m src.evaluation.visualize_volume \\
#         --image data/raw/BraTS-GLI-02062-102/BraTS-GLI-02062-102-t2w.nii \\
#         --seg data/raw/BraTS-GLI-02062-102/BraTS-GLI-02062-102-seg.nii
#
#     python -m src.evaluation.visualize_volume \\
#         --case BraTS-GLI-02062-102 --modality t2 --mode grid
# """
#
# import argparse
# import json
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import ListedColormap
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from scipy import ndimage
# from skimage import measure
#
# # Same legend as qualitative_figure.py, so figures from the two scripts
# # can be read against each other: NCR=turquoise, ED=violet, ET=yellow.
# CLASS_COLORS = {
#     0: (0, 0, 0, 0),
#     1: (0.25, 0.88, 0.82, 0.6),
#     2: (0.56, 0.0, 1.0, 0.5),
#     3: (1.0, 0.95, 0.0, 0.7),
# }
# CLASS_NAMES = {1: "NCR", 2: "ED", 3: "ET"}
#
# # Channel order written by src/datasets/prepare.py (MODALITY_ORDER).
# MODALITY_CHANNELS = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
#
# # This dataset (BraTS-2024-GLI) already uses contiguous raw labels
# # {0,1,2,3} directly (NCR=1, ED=2, ET=3) — confirmed against
# # src/datasets/prepare.py's own RAW_TO_CONTIGUOUS. That's DIFFERENT
# # from older BraTS (2017-2021), which used {1,2,4} with 3 skipped and
# # ET=4. Label 4 here means something else entirely: resection cavity
# # in some post-surgical cases, mapped to background rather than ET.
# RAW_LABEL_REMAP = {1: 1, 2: 2, 3: 3}
# RESECTION_CAVITY_RAW = 4
#
#
# def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
#     """Loads a .nii/.nii.gz via nibabel. Returns (volume, affine).
#
#     Imported lazily so the processed-.npy path still works in an
#     environment without nibabel installed."""
#     try:
#         import nibabel as nib
#     except ImportError as exc:  # pragma: no cover - depends on env
#         raise SystemExit(
#             "Reading .nii/.nii.gz needs nibabel: pip install nibabel\n"
#             "(not required when loading this project's processed .npy files)"
#         ) from exc
#
#     img = nib.load(str(path))
#     # get_fdata() applies the header's scl_slope/scl_inter scaling, which
#     # raw dataobj access skips — for intensity images that scaling is the
#     # difference between real and garbage values, so don't "optimize" it away.
#     return np.asarray(img.get_fdata(dtype=np.float32)), img.affine
#
#
# def load_segmentation(path: Path, raw_labels: bool) -> np.ndarray:
#     """Loads a segmentation from .nii/.nii.gz or .npy as integer labels.
#
#     Rounds rather than truncates on the float path: nibabel returns
#     floats, and an exact label of 3 can come back as 2.9999998 — this
#     is a confirmed issue on real BraTS-2024-GLI data (see prepare.py's
#     remap_labels), where int() would silently drop ET voxels to NCR."""
#     if path.suffix == ".npy":
#         seg = np.load(path)
#     else:
#         seg, _ = load_nifti(path)
#         raw_labels = True  # a .nii seg is always in raw BraTS label space
#     seg = np.rint(seg).astype(np.int16)
#
#     if raw_labels:
#         remapped = np.zeros_like(seg, dtype=np.uint8)
#         for raw_value, contiguous in RAW_LABEL_REMAP.items():
#             remapped[seg == raw_value] = contiguous
#         # Resection cavity: present in some post-surgical cases, explicitly
#         # background rather than an anomaly — matches prepare.py's handling.
#         unexpected = set(np.unique(seg)) - {0, RESECTION_CAVITY_RAW} - set(RAW_LABEL_REMAP)
#         if unexpected:
#             print(f"warning: ignoring unexpected raw label value(s) {sorted(unexpected)}")
#         return remapped
#
#     # imshow(vmin=0, vmax=3) would silently clamp an out-of-range label
#     # into ET's color, which is exactly what happens if a raw {1,2,4}
#     # segmentation is passed without --raw-labels: label 4 renders as
#     # yellow and looks plausible. Say so rather than showing a wrong picture.
#     out_of_range = sorted(int(v) for v in np.unique(seg) if v not in (0, 1, 2, 3))
#     if out_of_range:
#         print(
#             f"warning: label value(s) {out_of_range} outside the expected {{0,1,2,3}} range; "
#             "pass --raw-labels if this is a raw BraTS segmentation using {1,2,4}"
#         )
#     return seg.astype(np.uint8)
#
#
# def window_intensities(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
#     """Clips to a percentile window and rescales to [0, 1].
#
#     Without this, a handful of hyperintense outlier voxels compress all
#     the real tissue contrast into the bottom of the colormap and the
#     brain renders as a dim gray smear. Percentiles are taken over
#     nonzero voxels only, since BraTS volumes are skull-stripped and the
#     zero background otherwise dominates the low percentile."""
#     nonzero = volume[volume != 0]
#     if nonzero.size == 0:
#         return np.zeros_like(volume, dtype=np.float32)
#     lo, hi = np.percentile(nonzero, [low, high])
#     if hi <= lo:
#         lo, hi = float(nonzero.min()), float(nonzero.max())
#     if hi <= lo:
#         return np.zeros_like(volume, dtype=np.float32)
#     return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
#
#
# def pick_display_coords(volume: np.ndarray, seg: np.ndarray | None) -> tuple[int, int, int]:
#     """Shared cursor for the three orthogonal planes: the segmentation's
#     center of mass when there is one, else the center of the nonzero
#     (brain) region, which for a skull-stripped volume is a far more
#     useful default than the center of the padded array."""
#     if seg is not None and (seg > 0).any():
#         return tuple(int(round(c)) for c in np.argwhere(seg > 0).mean(axis=0))
#     if (volume != 0).any():
#         return tuple(int(round(c)) for c in np.argwhere(volume != 0).mean(axis=0))
#     return tuple(s // 2 for s in volume.shape)
#
#
# def _shade_faces(verts: np.ndarray, faces: np.ndarray, base_rgba: tuple, light_dir: np.ndarray) -> np.ndarray:
#     """Per-face Lambertian shading. Poly3DCollection paints every face a
#     single flat color by default, which is why an unlit marching-cubes
#     mesh reads as a flat silhouette rather than a solid object — there's
#     no cue for curvature. Dot each triangle's normal against a fixed
#     light direction and scale that face's color by the result."""
#     v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
#     normals = np.cross(v1 - v0, v2 - v0)
#     norms = np.linalg.norm(normals, axis=1, keepdims=True)
#     norms[norms == 0] = 1
#     normals = normals / norms
#     intensity = np.clip(normals @ light_dir, 0.0, 1.0)
#     intensity = 0.35 + 0.65 * intensity  # ambient floor so back faces aren't pure black
#     r, g, b, a = base_rgba
#     return np.column_stack([r * intensity, g * intensity, b * intensity, np.full_like(intensity, a)])
#
#
# def plot_3d_segmentation(ax, class_map: np.ndarray, downsample: int = 1, smooth_sigma: float = 1.0) -> None:
#     """Shaded 3D isosurface per tumor class (marching cubes on each
#     class's binary mask).
#
#     Gaussian-smoothing before thresholding merges the small disconnected
#     voxel islands a noisy mask produces, which would otherwise come out
#     as unreadable "dust"; it changes only the drawn surface, never any
#     computed metric. ED is drawn translucent and last, NCR/ET solid:
#     matplotlib depth-sorts whole collections rather than individual
#     triangles, so more than one nested translucent surface composites
#     incorrectly."""
#     light_dir = np.array([0.4, -0.5, 0.75])
#     light_dir = light_dir / np.linalg.norm(light_dir)
#     vol = class_map[::downsample, ::downsample, ::downsample]
#     drew_any = False
#     for cls in (1, 3, 2):  # solids first, translucent ED shell on top
#         mask = (vol == cls).astype(np.float32)
#         if mask.sum() < 15:
#             continue
#         smoothed = ndimage.gaussian_filter(mask, sigma=smooth_sigma)
#         if smoothed.max() < 0.5:
#             continue  # smoothing diluted an already-tiny fragment below the isolevel
#         try:
#             verts, faces, _, _ = measure.marching_cubes(smoothed, level=0.5)
#         except (ValueError, RuntimeError):
#             continue
#         r, g, b, _ = CLASS_COLORS[cls]
#         alpha = 0.35 if cls == 2 else 1.0
#         mesh = Poly3DCollection(verts[faces])
#         mesh.set_facecolor(_shade_faces(verts, faces, (r, g, b, alpha), light_dir))
#         mesh.set_edgecolor((0, 0, 0, 0))
#         ax.add_collection3d(mesh)
#         drew_any = True
#
#     ax.set_xlim(0, vol.shape[0])
#     ax.set_ylim(0, vol.shape[1])
#     ax.set_zlim(0, vol.shape[2])
#     ax.set_box_aspect(vol.shape)
#     # Panes on (rather than set_axis_off) give three receding walls —
#     # the depth cue that otherwise makes a render look like a 2D blob.
#     for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
#         axis.pane.set_facecolor((0.96, 0.96, 0.97, 1.0))
#         axis.pane.set_edgecolor((0.7, 0.7, 0.7, 1.0))
#         axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     ax.view_init(elev=22, azim=-55)
#     if not drew_any:
#         ax.text2D(0.5, 0.5, "no labelled\ntumor", ha="center", va="center", fontsize=8, transform=ax.transAxes)
#
#
# def export_interactive_3d(class_map: np.ndarray, out_path: Path, smooth_sigma: float = 1.0) -> None:
#     """Writes a self-contained HTML page with a rotatable/zoomable 3D
#     mesh via Plotly, referenced from its CDN — no `plotly` pip package
#     needed, this hand-builds the JSON Plotly.js expects. Same
#     smoothed-marching-cubes mesh as plot_3d_segmentation, just handed to
#     a real WebGL renderer with mouse-driven rotation instead of one
#     fixed matplotlib camera angle."""
#     traces = []
#     for cls in (1, 2, 3):
#         mask = (class_map == cls).astype(np.float32)
#         if mask.sum() < 15:
#             continue
#         smoothed = ndimage.gaussian_filter(mask, sigma=smooth_sigma)
#         if smoothed.max() < 0.5:
#             continue
#         try:
#             verts, faces, _, _ = measure.marching_cubes(smoothed, level=0.5)
#         except (ValueError, RuntimeError):
#             continue
#         r, g, b, _ = CLASS_COLORS[cls]
#         traces.append(
#             {
#                 "type": "mesh3d",
#                 "x": verts[:, 0].tolist(),
#                 "y": verts[:, 1].tolist(),
#                 "z": verts[:, 2].tolist(),
#                 "i": faces[:, 0].tolist(),
#                 "j": faces[:, 1].tolist(),
#                 "k": faces[:, 2].tolist(),
#                 "color": f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})",
#                 "opacity": 0.35 if cls == 2 else 0.95,  # ED translucent shell, NCR/ET solid — same reasoning as the static render
#                 "flatshading": False,
#                 "lighting": {"ambient": 0.45, "diffuse": 0.8, "specular": 0.35, "roughness": 0.5, "fresnel": 0.1},
#                 "lightposition": {"x": 200, "y": 200, "z": 300},
#                 "name": CLASS_NAMES[cls],
#                 "showscale": False,
#             }
#         )
#
#     if not traces:
#         print(f"No tumor mesh to export for {out_path.name} (segmentation is empty) — skipping.")
#         return
#
#     layout = {
#         "scene": {
#             "xaxis": {"visible": False},
#             "yaxis": {"visible": False},
#             "zaxis": {"visible": False},
#             "aspectmode": "data",  # true relative proportions, not a cube-stretched shape
#         },
#         "paper_bgcolor": "white",
#         "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
#         "showlegend": True,
#     }
#     html = f"""<!DOCTYPE html>
# <html><head><meta charset="utf-8"><title>{out_path.stem}</title></head>
# <body style="margin:0">
# <div id="plot" style="width:100vw;height:100vh;"></div>
# <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
# <script>
# Plotly.newPlot('plot', {json.dumps(traces)}, {json.dumps(layout)}, {{responsive: true}});
# </script>
# </body></html>"""
#     out_path.write_text(html, encoding="utf-8")
#     print(f"Wrote {out_path} (open in a browser — drag to rotate, scroll to zoom)")
#
#
# def orthogonal_panels(volume: np.ndarray, seg: np.ndarray | None, coords: tuple[int, int, int]) -> list:
#     """(title, base_slice, seg_slice) for the three orthogonal planes
#     through `coords`. Coronal/sagittal are transposed so the D axis runs
#     vertically in every panel, as in a tri-planar viewer."""
#     h, w, d = coords
#     panels = [
#         ("Axial", volume[:, :, d], None if seg is None else seg[:, :, d]),
#         ("Coronal", volume[h, :, :].T, None if seg is None else seg[h, :, :].T),
#         ("Sagittal", volume[:, w, :].T, None if seg is None else seg[:, w, :].T),
#     ]
#     return panels
#
#
# def add_legend(fig, seg: np.ndarray) -> None:
#     """Legend for whichever classes are actually present, so a case with
#     no enhancing tumor doesn't advertise a color that never appears."""
#     present = [c for c in (1, 2, 3) if (seg == c).any()]
#     if not present:
#         return
#     handles = [
#         plt.Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=CLASS_COLORS[c], label=CLASS_NAMES[c])
#         for c in present
#     ]
#     fig.legend(handles=handles, loc="lower center", ncol=len(present), fontsize=9, bbox_to_anchor=(0.5, -0.01))
#
#
# def build_triplanar(volume: np.ndarray, seg: np.ndarray | None, title: str, out_path: Path) -> None:
#     coords = pick_display_coords(volume, seg)
#     panels = orthogonal_panels(volume, seg, coords)
#     cmap = ListedColormap([CLASS_COLORS[i] for i in range(4)])
#
#     n_cols = 4 if seg is not None else 3
#     fig = plt.figure(figsize=(3.2 * n_cols, 3.6))
#     for col, (name, base_slice, seg_slice) in enumerate(panels):
#         ax = fig.add_subplot(1, n_cols, col + 1)
#         ax.imshow(base_slice, cmap="gray", vmin=0, vmax=1)
#         if seg_slice is not None:
#             ax.imshow(seg_slice, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
#         ax.set_title(name, fontsize=10)
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     if seg is not None:
#         ax3d = fig.add_subplot(1, n_cols, 4, projection="3d")
#         plot_3d_segmentation(ax3d, seg)
#         ax3d.set_title("3D render", fontsize=10)
#         add_legend(fig, seg)
#
#     fig.suptitle(f"{title}  |  cursor {coords}  |  shape {volume.shape}", fontsize=11)
#     fig.tight_layout(rect=(0, 0.03, 1, 0.96))
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     print(f"Wrote {out_path}")
#
#
# def build_grid(volume: np.ndarray, seg: np.ndarray | None, title: str, out_path: Path, n_slices: int) -> None:
#     """Montage of evenly spaced axial slices across the brain's extent.
#
#     Spans only the slices that actually contain signal rather than the
#     full array: BraTS volumes are zero-padded at both ends, so an even
#     spread over the whole D axis wastes several panels on blank slices."""
#     nonzero_slices = np.flatnonzero((volume != 0).any(axis=(0, 1)))
#     if nonzero_slices.size == 0:
#         lo, hi = 0, volume.shape[2] - 1
#     else:
#         lo, hi = int(nonzero_slices[0]), int(nonzero_slices[-1])
#     indices = np.unique(np.linspace(lo, hi, n_slices).round().astype(int))
#
#     cmap = ListedColormap([CLASS_COLORS[i] for i in range(4)])
#     n_cols = min(6, len(indices))
#     n_rows = int(np.ceil(len(indices) / n_cols))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows), squeeze=False)
#
#     for ax, idx in zip(axes.ravel(), indices):
#         ax.imshow(volume[:, :, idx], cmap="gray", vmin=0, vmax=1)
#         if seg is not None:
#             ax.imshow(seg[:, :, idx], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
#         ax.set_title(f"z={idx}", fontsize=8)
#     for ax in axes.ravel():
#         ax.set_xticks([])
#         ax.set_yticks([])
#     for ax in axes.ravel()[len(indices) :]:
#         ax.set_visible(False)  # trailing cells when the count doesn't fill the grid
#
#     if seg is not None:
#         add_legend(fig, seg)
#     fig.suptitle(f"{title}  |  axial slices {lo}-{hi}  |  shape {volume.shape}", fontsize=11)
#     fig.tight_layout(rect=(0, 0.03, 1, 0.95))
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     print(f"Wrote {out_path}")
#
#
# def resolve_inputs(args) -> tuple[np.ndarray, np.ndarray | None, str]:
#     """Resolves --image/--seg or --case into (volume, seg, title)."""
#     if args.case:
#         case_dir = Path(args.processed_dir) / args.case
#         image_path = case_dir / "image.npy"
#         if not image_path.exists():
#             raise SystemExit(f"No processed image at {image_path}")
#         stacked = np.load(image_path)  # (4, H, W, D)
#         volume = stacked[MODALITY_CHANNELS[args.modality]]
#         seg_path = case_dir / "seg.npy"
#         # Processed seg.npy is already in contiguous {0,1,2,3} label space.
#         seg = load_segmentation(seg_path, raw_labels=False) if seg_path.exists() else None
#         return volume, seg, f"{args.case} ({args.modality}, processed)"
#
#     image_path = Path(args.image)
#     if not image_path.exists():
#         raise SystemExit(f"No such file: {image_path}")
#     if image_path.suffix == ".npy":
#         volume = np.load(image_path)
#         if volume.ndim == 4:  # a full (4, H, W, D) image.npy rather than one modality
#             volume = volume[MODALITY_CHANNELS[args.modality]]
#     else:
#         volume, _ = load_nifti(image_path)
#
#     seg = load_segmentation(Path(args.seg), args.raw_labels) if args.seg else None
#     return volume, seg, image_path.name
#
#
# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     source = parser.add_mutually_exclusive_group(required=True)
#     source.add_argument("--image", help="Path to a .nii/.nii.gz volume, or a processed .npy array")
#     source.add_argument("--case", help="Case ID to load from the processed data directory")
#     parser.add_argument("--seg", help="Optional segmentation (.nii/.nii.gz/.npy) to overlay")
#     parser.add_argument("--processed-dir", default="data/processed", help="Processed data root (for --case)")
#     parser.add_argument(
#         "--modality",
#         default="t2",
#         choices=sorted(MODALITY_CHANNELS),
#         help="Which channel to display from a 4-channel image.npy",
#     )
#     parser.add_argument("--mode", default="triplanar", choices=["triplanar", "grid"])
#     parser.add_argument("--n-slices", type=int, default=12, help="Number of slices in --mode grid")
#     parser.add_argument(
#         "--raw-labels",
#         action="store_true",
#         help="Treat a .npy segmentation as raw labels (resection cavity=4 -> background) "
#         "(automatic for .nii input)",
#     )
#     parser.add_argument("--out", help="Output image path (default: derived from the input name)")
#     args = parser.parse_args()
#
#     volume, seg, title = resolve_inputs(args)
#     if seg is not None and seg.shape != volume.shape:
#         raise SystemExit(f"Segmentation shape {seg.shape} does not match image shape {volume.shape}")
#
#     volume = window_intensities(volume)
#
#     if args.out:
#         out_path = Path(args.out)
#     else:
#         stem = args.case if args.case else Path(args.image).name.split(".")[0]
#         out_path = Path(f"{stem}_{args.mode}.png")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     if args.mode == "triplanar":
#         build_triplanar(volume, seg, title, out_path)
#     else:
#         build_grid(volume, seg, title, out_path, args.n_slices)
#
#     if seg is not None:
#         export_interactive_3d(seg, out_path.with_name(out_path.stem + "_3d.html"))
#
#
# if __name__ == "__main__":
#     main()


"""Standalone viewer for a single case's MRI volume — raw .nii/.nii.gz
(loaded with nibabel) or this project's processed .npy arrays.

This is a debugging/inspection tool, deliberately separate from
qualitative_figure.py: that script re-runs ensemble inference to render
*model predictions* for the 5 percentile-selected holdout cases, whereas
this one just looks at the data itself (any modality, any case, with or
without the ground-truth segmentation) and needs no checkpoints, no GPU,
and no eval_results_*.json.

Two display modes, both following the standard convention of a
color-coded semi-transparent overlay on a grayscale base image:

  --mode triplanar  (default) axial/coronal/sagittal through one shared
                    cursor — the ITK-SNAP 4-pane layout, plus a 3D
                    isosurface render when a segmentation is available.
  --mode grid       a montage of evenly-spaced axial slices, for
                    scanning a whole volume at once.

Raw vs processed inputs differ in ways that matter here:

  raw .nii       original intensities, original (H, W, D) shape, raw
                 BraTS label values {0,1,2,4}, and a real affine.
  processed .npy image.npy is (4, H, W, D) z-scored per modality in the
                 fixed order t1, t1ce, t2, flair; seg.npy holds labels
                 already remapped from raw BraTS-2024-GLI values (see
                 below) to contiguous {0,1,2,3}; the affine is NOT
                 saved, so orientation can't be recovered from it.

This dataset (BraTS-2024-GLI) already labels segmentations {0,1,2,3}
directly (NCR=1, ED=2, ET=3) — NOT the {1,2,4} scheme older BraTS
(2017-2021) used. Raw label 4 here means resection cavity, present in
some post-surgical cases, and is treated as background, matching
src/datasets/prepare.py's own remap_labels(). --raw-labels applies
that same handling to a .npy segmentation; it's automatic for .nii
input, since a .nii segmentation is always in raw label space.

Whenever a segmentation is given, this also writes a companion
<name>_3d.html next to the PNG: a rotatable/zoomable 3D mesh (Plotly,
loaded from a CDN — needs internet the first time it's opened, nothing
else, no pip install). This exists because a static render can only
ever approximate depth with one fixed camera angle and fixed lighting —
real anatomy (edema is often a thin curved sheet, not a round blob) can
look flat from that one angle no matter how it's lit. Letting you
actually drag to rotate it removes that ambiguity.

Batch mode: pass more than one --case, or --all-cases to sweep every
case under --processed-dir. Each case's outputs go to --out-dir (default:
current directory), named by case ID — --out only makes sense for a
single case. One case failing (missing seg.npy, shape mismatch, etc.)
is reported and skipped rather than aborting the whole run; see the
summary line printed at the end.

Examples:
    python -m src.evaluation.visualize_volume \\
        --image data/raw/BraTS-GLI-02062-102/BraTS-GLI-02062-102-t2w.nii

    python -m src.evaluation.visualize_volume \\
        --image data/raw/BraTS-GLI-02062-102/BraTS-GLI-02062-102-t2w.nii \\
        --seg data/raw/BraTS-GLI-02062-102/BraTS-GLI-02062-102-seg.nii

    python -m src.evaluation.visualize_volume \\
        --case BraTS-GLI-02062-102 --modality t2 --mode grid

    python -m src.evaluation.visualize_volume \\
        --case BraTS-GLI-02062-102 BraTS-GLI-02092-103 --out-dir figs/

    python -m src.evaluation.visualize_volume --all-cases --out-dir figs/
"""

import argparse
import json
from pathlib import Path
import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage import measure

# Same legend as qualitative_figure.py, so figures from the two scripts
# can be read against each other: NCR=turquoise, ED=violet, ET=yellow.
CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (0.25, 0.88, 0.82, 0.6),
    2: (0.56, 0.0, 1.0, 0.5),
    3: (1.0, 0.95, 0.0, 0.7),
}
CLASS_NAMES = {1: "NCR", 2: "ED", 3: "ET"}

# Channel order written by src/datasets/prepare.py (MODALITY_ORDER).
MODALITY_CHANNELS = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}

# This dataset (BraTS-2024-GLI) already uses contiguous raw labels
# {0,1,2,3} directly (NCR=1, ED=2, ET=3) — confirmed against
# src/datasets/prepare.py's own RAW_TO_CONTIGUOUS. That's DIFFERENT
# from older BraTS (2017-2021), which used {1,2,4} with 3 skipped and
# ET=4. Label 4 here means something else entirely: resection cavity
# in some post-surgical cases, mapped to background rather than ET.
RAW_LABEL_REMAP = {1: 1, 2: 2, 3: 3}
RESECTION_CAVITY_RAW = 4


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Loads a .nii/.nii.gz via nibabel. Returns (volume, affine).

    Imported lazily so the processed-.npy path still works in an
    environment without nibabel installed."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on env
        raise SystemExit(
            "Reading .nii/.nii.gz needs nibabel: pip install nibabel\n"
            "(not required when loading this project's processed .npy files)"
        ) from exc

    
    img = typing.cast(nib.Nifti1Image, nib.load(str(path)))
    # get_fdata() applies the header's scl_slope/scl_inter scaling, which
    # raw dataobj access skips — for intensity images that scaling is the
    # difference between real and garbage values, so don't "optimize" it away.
    return np.asarray(img.get_fdata(dtype=np.float32)), img.affine


def load_segmentation(path: Path, raw_labels: bool) -> np.ndarray:
    """Loads a segmentation from .nii/.nii.gz or .npy as integer labels.

    Rounds rather than truncates on the float path: nibabel returns
    floats, and an exact label of 3 can come back as 2.9999998 — this
    is a confirmed issue on real BraTS-2024-GLI data (see prepare.py's
    remap_labels), where int() would silently drop ET voxels to NCR."""
    if path.suffix == ".npy":
        seg = np.load(path)
    else:
        seg, _ = load_nifti(path)
        raw_labels = True  # a .nii seg is always in raw BraTS label space
    seg = np.rint(seg).astype(np.int16)

    if raw_labels:
        remapped = np.zeros_like(seg, dtype=np.uint8)
        for raw_value, contiguous in RAW_LABEL_REMAP.items():
            remapped[seg == raw_value] = contiguous
        # Resection cavity: present in some post-surgical cases, explicitly
        # background rather than an anomaly — matches prepare.py's handling.
        unexpected = set(np.unique(seg)) - {0, RESECTION_CAVITY_RAW} - set(RAW_LABEL_REMAP)
        if unexpected:
            print(f"warning: ignoring unexpected raw label value(s) {sorted(unexpected)}")
        return remapped

    # imshow(vmin=0, vmax=3) would silently clamp an out-of-range label
    # into ET's color, which is exactly what happens if a raw {1,2,4}
    # segmentation is passed without --raw-labels: label 4 renders as
    # yellow and looks plausible. Say so rather than showing a wrong picture.
    out_of_range = sorted(int(v) for v in np.unique(seg) if v not in (0, 1, 2, 3))
    if out_of_range:
        print(
            f"warning: label value(s) {out_of_range} outside the expected {{0,1,2,3}} range; "
            "pass --raw-labels if this is a raw BraTS segmentation using {1,2,4}"
        )
    return seg.astype(np.uint8)


def window_intensities(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clips to a percentile window and rescales to [0, 1].

    Without this, a handful of hyperintense outlier voxels compress all
    the real tissue contrast into the bottom of the colormap and the
    brain renders as a dim gray smear. Percentiles are taken over
    nonzero voxels only, since BraTS volumes are skull-stripped and the
    zero background otherwise dominates the low percentile."""
    nonzero = volume[volume != 0]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(nonzero, [low, high])
    if hi <= lo:
        lo, hi = float(nonzero.min()), float(nonzero.max())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def pick_display_coords(volume: np.ndarray, seg: np.ndarray | None) -> tuple[int, int, int]:
    """Shared cursor for the three orthogonal planes: the segmentation's
    center of mass when there is one, else the center of the nonzero
    (brain) region, which for a skull-stripped volume is a far more
    useful default than the center of the padded array."""
    if seg is not None and (seg > 0).any():
        return tuple(round(c) for c in np.argwhere(seg > 0).mean(axis=0))
    if (volume != 0).any():
        return tuple(round(c) for c in np.argwhere(volume != 0).mean(axis=0))
    return tuple(s // 2 for s in volume.shape)


def _shade_faces(verts: np.ndarray, faces: np.ndarray, base_rgba: tuple, light_dir: np.ndarray) -> np.ndarray:
    """Per-face Lambertian shading. Poly3DCollection paints every face a
    single flat color by default, which is why an unlit marching-cubes
    mesh reads as a flat silhouette rather than a solid object — there's
    no cue for curvature. Dot each triangle's normal against a fixed
    light direction and scale that face's color by the result."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    intensity = np.clip(normals @ light_dir, 0.0, 1.0)
    intensity = 0.35 + 0.65 * intensity  # ambient floor so back faces aren't pure black
    r, g, b, a = base_rgba
    return np.column_stack([r * intensity, g * intensity, b * intensity, np.full_like(intensity, a)])


def plot_3d_segmentation(ax, class_map: np.ndarray, downsample: int = 1, smooth_sigma: float = 1.0) -> None:
    """Shaded 3D isosurface per tumor class (marching cubes on each
    class's binary mask).

    Gaussian-smoothing before thresholding merges the small disconnected
    voxel islands a noisy mask produces, which would otherwise come out
    as unreadable "dust"; it changes only the drawn surface, never any
    computed metric. ED is drawn translucent and last, NCR/ET solid:
    matplotlib depth-sorts whole collections rather than individual
    triangles, so more than one nested translucent surface composites
    incorrectly."""
    light_dir = np.array([0.4, -0.5, 0.75])
    light_dir = light_dir / np.linalg.norm(light_dir)
    vol = class_map[::downsample, ::downsample, ::downsample]
    drew_any = False
    for cls in (1, 3, 2):  # solids first, translucent ED shell on top
        mask = (vol == cls).astype(np.float32)
        if mask.sum() < 15:
            continue
        smoothed = ndimage.gaussian_filter(mask, sigma=smooth_sigma)
        if smoothed.max() < 0.5:
            continue  # smoothing diluted an already-tiny fragment below the isolevel
        try:
            verts, faces, _, _ = measure.marching_cubes(smoothed, level=0.5)
        except (ValueError, RuntimeError):
            continue
        r, g, b, _ = CLASS_COLORS[cls]
        alpha = 0.35 if cls == 2 else 1.0
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor(_shade_faces(verts, faces, (r, g, b, alpha), light_dir))
        mesh.set_edgecolor((0, 0, 0, 0))
        ax.add_collection3d(mesh)
        drew_any = True

    ax.set_xlim(0, vol.shape[0])
    ax.set_ylim(0, vol.shape[1])
    ax.set_zlim(0, vol.shape[2])
    ax.set_box_aspect(vol.shape)
    # Panes on (rather than set_axis_off) give three receding walls —
    # the depth cue that otherwise makes a render look like a 2D blob.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.96, 0.97, 1.0))
        axis.pane.set_edgecolor((0.7, 0.7, 0.7, 1.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-55)
    if not drew_any:
        ax.text2D(0.5, 0.5, "no labelled\ntumor", ha="center", va="center", fontsize=8, transform=ax.transAxes)


def export_interactive_3d(class_map: np.ndarray, out_path: Path, smooth_sigma: float = 1.0) -> None:
    """Writes a self-contained HTML page with a rotatable/zoomable 3D
    mesh via Plotly, referenced from its CDN — no `plotly` pip package
    needed, this hand-builds the JSON Plotly.js expects. Same
    smoothed-marching-cubes mesh as plot_3d_segmentation, just handed to
    a real WebGL renderer with mouse-driven rotation instead of one
    fixed matplotlib camera angle."""
    traces = []
    for cls in (1, 2, 3):
        mask = (class_map == cls).astype(np.float32)
        if mask.sum() < 15:
            continue
        smoothed = ndimage.gaussian_filter(mask, sigma=smooth_sigma)
        if smoothed.max() < 0.5:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(smoothed, level=0.5)
        except (ValueError, RuntimeError):
            continue
        r, g, b, _ = CLASS_COLORS[cls]
        traces.append(
            {
                "type": "mesh3d",
                "x": verts[:, 0].tolist(),
                "y": verts[:, 1].tolist(),
                "z": verts[:, 2].tolist(),
                "i": faces[:, 0].tolist(),
                "j": faces[:, 1].tolist(),
                "k": faces[:, 2].tolist(),
                "color": f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})",
                "opacity": (
                    0.35 if cls == 2 else 0.95
                ),  # ED translucent shell, NCR/ET solid — same reasoning as the static render
                "flatshading": False,
                "lighting": {"ambient": 0.45, "diffuse": 0.8, "specular": 0.35, "roughness": 0.5, "fresnel": 0.1},
                "lightposition": {"x": 200, "y": 200, "z": 300},
                "name": CLASS_NAMES[cls],
                "showscale": False,
            }
        )

    if not traces:
        print(f"No tumor mesh to export for {out_path.name} (segmentation is empty) — skipping.")
        return

    layout = {
        "scene": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",  # true relative proportions, not a cube-stretched shape
        },
        "paper_bgcolor": "white",
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "showlegend": True,
    }
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{out_path.stem}</title></head>
<body style="margin:0">
<div id="plot" style="width:100vw;height:100vh;"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
Plotly.newPlot('plot', {json.dumps(traces)}, {json.dumps(layout)}, {{responsive: true}});
</script>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} (open in a browser — drag to rotate, scroll to zoom)")


def orthogonal_panels(volume: np.ndarray, seg: np.ndarray | None, coords: tuple[int, int, int]) -> list:
    """(title, base_slice, seg_slice) for the three orthogonal planes
    through `coords`. Coronal/sagittal are transposed so the D axis runs
    vertically in every panel, as in a tri-planar viewer."""
    h, w, d = coords
    panels = [
        ("Axial", volume[:, :, d], None if seg is None else seg[:, :, d]),
        ("Coronal", volume[h, :, :].T, None if seg is None else seg[h, :, :].T),
        ("Sagittal", volume[:, w, :].T, None if seg is None else seg[:, w, :].T),
    ]
    return panels


def add_legend(fig, seg: np.ndarray) -> None:
    """Legend for whichever classes are actually present, so a case with
    no enhancing tumor doesn't advertise a color that never appears."""
    present = [c for c in (1, 2, 3) if (seg == c).any()]
    if not present:
        return
    handles = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=CLASS_COLORS[c], label=CLASS_NAMES[c]
        )
        for c in present
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(present), fontsize=9, bbox_to_anchor=(0.5, -0.01))


def build_triplanar(volume: np.ndarray, seg: np.ndarray | None, title: str, out_path: Path) -> None:
    coords = pick_display_coords(volume, seg)
    panels = orthogonal_panels(volume, seg, coords)
    cmap = ListedColormap([CLASS_COLORS[i] for i in range(4)])

    n_cols = 4 if seg is not None else 3
    fig = plt.figure(figsize=(3.2 * n_cols, 3.6))
    for col, (name, base_slice, seg_slice) in enumerate(panels):
        ax = fig.add_subplot(1, n_cols, col + 1)
        ax.imshow(base_slice, cmap="gray", vmin=0, vmax=1)
        if seg_slice is not None:
            ax.imshow(seg_slice, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if seg is not None:
        ax3d = fig.add_subplot(1, n_cols, 4, projection="3d")
        plot_3d_segmentation(ax3d, seg)
        ax3d.set_title("3D render", fontsize=10)
        add_legend(fig, seg)

    fig.suptitle(f"{title}  |  cursor {coords}  |  shape {volume.shape}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def build_grid(volume: np.ndarray, seg: np.ndarray | None, title: str, out_path: Path, n_slices: int) -> None:
    """Montage of evenly spaced axial slices across the brain's extent.

    Spans only the slices that actually contain signal rather than the
    full array: BraTS volumes are zero-padded at both ends, so an even
    spread over the whole D axis wastes several panels on blank slices."""
    nonzero_slices = np.flatnonzero((volume != 0).any(axis=(0, 1)))
    if nonzero_slices.size == 0:
        lo, hi = 0, volume.shape[2] - 1
    else:
        lo, hi = int(nonzero_slices[0]), int(nonzero_slices[-1])
    indices = np.unique(np.linspace(lo, hi, n_slices).round().astype(int))

    cmap = ListedColormap([CLASS_COLORS[i] for i in range(4)])
    n_cols = min(6, len(indices))
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows), squeeze=False)

    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(volume[:, :, idx], cmap="gray", vmin=0, vmax=1)
        if seg is not None:
            ax.imshow(seg[:, :, idx], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"z={idx}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(indices) :]:
        ax.set_visible(False)  # trailing cells when the count doesn't fill the grid

    if seg is not None:
        add_legend(fig, seg)
    fig.suptitle(f"{title}  |  axial slices {lo}-{hi}  |  shape {volume.shape}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def load_processed_case(case_id: str, processed_dir: Path, modality: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Loads one case's (volume, seg, title) from the processed data
    directory. Raises FileNotFoundError (not SystemExit) so batch mode
    can catch it per-case and keep going rather than aborting the whole
    sweep on one missing case."""
    case_dir = processed_dir / case_id
    image_path = case_dir / "image.npy"
    if not image_path.exists():
        raise FileNotFoundError(f"no processed image at {image_path}")
    stacked = np.load(image_path)  # (4, H, W, D)
    volume = stacked[MODALITY_CHANNELS[modality]]
    seg_path = case_dir / "seg.npy"
    # Processed seg.npy is already in contiguous {0,1,2,3} label space.
    seg = load_segmentation(seg_path, raw_labels=False) if seg_path.exists() else None
    return volume, seg, f"{case_id} ({modality}, processed)"


def resolve_image_input(args) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Resolves --image/--seg (the single-file, non-batch path)."""
    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"No such file: {image_path}")
    if image_path.suffix == ".npy":
        volume = np.load(image_path)
        if volume.ndim == 4:  # a full (4, H, W, D) image.npy rather than one modality
            volume = volume[MODALITY_CHANNELS[args.modality]]
    else:
        volume, _ = load_nifti(image_path)

    seg = load_segmentation(Path(args.seg), args.raw_labels) if args.seg else None
    return volume, seg, image_path.name


def process_one(volume: np.ndarray, seg: np.ndarray | None, title: str, out_path: Path, args) -> None:
    """Shape-check, window, render, and (if a segmentation is present)
    export the interactive 3D HTML for one case. Raises ValueError on a
    shape mismatch — the single-case path in main() lets that abort the
    run via SystemExit; the batch path catches it and skips just that
    case, matching how load_processed_case's FileNotFoundError is handled."""
    if seg is not None and seg.shape != volume.shape:
        raise ValueError(f"segmentation shape {seg.shape} does not match image shape {volume.shape}")

    volume = window_intensities(volume)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "triplanar":
        build_triplanar(volume, seg, title, out_path)
    else:
        build_grid(volume, seg, title, out_path, args.n_slices)

    if seg is not None:
        export_interactive_3d(seg, out_path.with_name(out_path.stem + "_3d.html"))


def resolve_case_ids(args) -> list[str]:
    """--case (one or more explicit IDs) or --all-cases (every case
    directory under --processed-dir that has an image.npy)."""
    if args.all_cases:
        processed_dir = Path(args.processed_dir)
        if not processed_dir.exists():
            raise SystemExit(f"No such directory: {processed_dir}")
        case_ids = sorted(p.name for p in processed_dir.iterdir() if p.is_dir() and (p / "image.npy").exists())
        if not case_ids:
            raise SystemExit(f"No cases with image.npy found under {processed_dir}")
        return case_ids
    return args.case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a .nii/.nii.gz volume, or a processed .npy array")
    source.add_argument("--case", nargs="+", help="One or more case IDs to load from the processed data directory")
    source.add_argument("--all-cases", action="store_true", help="Process every case found under --processed-dir")
    parser.add_argument("--seg", help="Optional segmentation (.nii/.nii.gz/.npy) to overlay (single-case --image only)")
    parser.add_argument(
        "--processed-dir", default="data/processed", help="Processed data root (for --case/--all-cases)"
    )
    parser.add_argument(
        "--modality",
        default="t2",
        choices=sorted(MODALITY_CHANNELS),
        help="Which channel to display from a 4-channel image.npy",
    )
    parser.add_argument("--mode", default="triplanar", choices=["triplanar", "grid"])
    parser.add_argument("--n-slices", type=int, default=12, help="Number of slices in --mode grid")
    parser.add_argument(
        "--raw-labels",
        action="store_true",
        help="Treat a .npy segmentation as raw labels (resection cavity=4 -> background) " "(automatic for .nii input)",
    )
    parser.add_argument("--out", help="Output image path — single case only")
    parser.add_argument("--out-dir", help="Output directory for batch mode (default: current directory)")
    args = parser.parse_args()

    if args.image:
        if args.out_dir:
            raise SystemExit("--out-dir is for batch mode (--case with several IDs, or --all-cases); use --out")
        volume, seg, title = resolve_image_input(args)
        stem = Path(args.image).name.split(".")[0]
        out_path = Path(args.out) if args.out else Path(f"{stem}_{args.mode}.png")
        try:
            process_one(volume, seg, title, out_path, args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return

    case_ids = resolve_case_ids(args)
    batch = len(case_ids) > 1
    if batch and args.out:
        raise SystemExit("--out is for a single case; use --out-dir for multiple cases")
    out_dir = Path(args.out_dir) if args.out_dir else Path(".")
    processed_dir = Path(args.processed_dir)

    succeeded, failed = [], []
    for case_id in case_ids:
        try:
            volume, seg, title = load_processed_case(case_id, processed_dir, args.modality)
            out_path = Path(args.out) if (args.out and not batch) else out_dir / f"{case_id}_{args.mode}.png"
            process_one(volume, seg, title, out_path, args)
            succeeded.append(case_id)
        except (FileNotFoundError, ValueError) as exc:
            print(f"skipping {case_id}: {exc}")
            failed.append(case_id)

    if batch:
        print(
            f"\n{len(succeeded)}/{len(case_ids)} cases written to {out_dir}" + (f"; failed: {failed}" if failed else "")
        )


if __name__ == "__main__":
    main()
