"""Ensembling across CV folds. Matches the paper's own description
(Section 3.2): "ensembling was implemented by first predicting the test
cases individually with each configuration, followed by averaging the
sigmoid outputs to obtain the final prediction" — for the region-based
(sigmoid) variants. For softmax-mode variants (BL, BL*), this uses the
standard nnU-Net/soft-voting equivalent: averaging the per-class softmax
probabilities across models before argmax. Both are the same operation
(elementwise average of K models' probability-valued outputs) and both
compose correctly with pred_to_regions(), which already branches on
region_based_training to interpret the averaged tensor correctly either
way — there was never a real reason to reject softmax mode here.

Two fixes relative to the plan's original snippet, both found by
actually trying to run it against this project's real architecture:
  1. `model_cls.load_from_checkpoint(...)` is PyTorch Lightning API —
     doesn't exist on a plain nn.Module. Checkpoints here are raw
     state_dicts (torch.save(model.state_dict(), ...), Phase 6), so
     loading means instantiating the model from cfg.model first, then
     load_state_dict().
  2. NNUNet3D's SegmentationHead already applies softmax/sigmoid
     internally (Phase 1) — forward() returns probabilities, never raw
     logits. Re-applying torch.sigmoid() here would double-apply the
     nonlinearity. No extra nonlinearity is applied in this file.

A third issue found via a real Phase 9 run: this function originally
raised ValueError for non-region-based (softmax) models, on the
reasoning that "averaging mutually-exclusive-class probabilities isn't
the same operation as averaging independent per-region probabilities."
That's true in the sense that the numbers mean something different, but
it doesn't make averaging softmax probabilities invalid — it's the
standard ensembling method nnU-Net itself uses, and pred_to_regions()
already interprets the result correctly for either mode. The
restriction blocked 2 of 8 real ablation variants (baseline,
baseline_bs5) from ever getting a Table 2 result at all. Removed.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig


def ensemble_predict(
    checkpoint_paths: list[str], cfg: DictConfig, batch: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Loads each checkpoint into a freshly-instantiated model (matching
    cfg.model), runs a forward pass, averages the resulting
    probabilities. Uses only the final (full-resolution) output from
    each model's deep-supervision output list.

    Works for both region-based (sigmoid) and softmax-mode models —
    averaging probability-valued outputs elementwise is the same
    operation either way, and the caller's pred_to_regions() already
    interprets the averaged tensor correctly depending on
    cfg.model.region_based_training.
    """
    total: torch.Tensor | None = None
    for ckpt_path in checkpoint_paths:
        model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
        state_dict = torch.load(Path(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            outputs = model(batch.to(device))  # list of deep-supervision outputs
            final_output = outputs[-1]  # full-resolution output, already probability-valued
        total = final_output if total is None else total + final_output

    assert total is not None, "checkpoint_paths must be non-empty"
    return total / len(checkpoint_paths)
