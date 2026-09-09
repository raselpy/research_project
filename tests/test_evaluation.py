import pytest

from src.evaluation.run import _fold_from_checkpoint_path


def test_fold_parsed_correctly_from_checkpoint_path():
    """Regression test for a real train/val leakage bug: the original
    cv_val evaluation scored every checkpoint against ALL train_pool
    cases (every case that was SOME fold's held-out set), not just the
    cases its OWN fold held out — meaning a fold-0 checkpoint was
    evaluated on cases it had actually trained on, producing
    artificially inflated Dice/HD95. Fixed by parsing the fold number
    from the checkpoint path and scoring only that fold's own held-out
    cases."""
    assert _fold_from_checkpoint_path("results/checkpoints/baseline/fold_0.pt") == 0
    assert _fold_from_checkpoint_path("results/checkpoints/baseline/fold_4.pt") == 4
    assert _fold_from_checkpoint_path(r"results\checkpoints\baseline\fold_2.pt") == 2


def test_fold_parsing_raises_on_unrecognized_path():
    with pytest.raises(ValueError):
        _fold_from_checkpoint_path("results/checkpoints/baseline/latest.pt")
