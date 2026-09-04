"""Ensembling across CV folds. Matches the paper's own description
exactly (Section 3.2): "ensembling was implemented by first predicting
the test cases individually with each configuration, followed by
averaging the sigmoid outputs to obtain the final prediction."

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

    Requires cfg.model.region_based_training=True: the paper's
    ensembling method averages independent per-region sigmoid
    probabilities. Softmax-mode models produce probabilities over
    mutually exclusive classes instead — naively averaging those isn't
    the same operation and isn't what the paper describes, so this
    raises rather than silently producing a result that looks
    reasonable but isn't what was asked for.
    """
    if not cfg.model.region_based_training:
        raise ValueError(
            "ensemble_predict requires cfg.model.region_based_training=True — "
            "the paper's ensembling method averages independent per-region "
            "sigmoid probabilities, not softmax outputs over mutually "
            "exclusive classes. Use the region-based (R) ablation's "
            "checkpoints for ensembling, or evaluate softmax-mode models "
            "individually instead."
        )

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