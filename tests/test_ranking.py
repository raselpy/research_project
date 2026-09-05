import pandas as pd

from src.evaluation.ranking import compute_brats_ranking


def test_brats_ranking_hand_computable_example():
    """The plan's own acceptance check (Phase 8, 8.3): 2 models, 2 cases,
    only whole_dice varies (others tied) — model A strictly dominates on
    whole_dice and must rank better (lower score).

    By hand: A's per-case ranks = [1, 1.5, 1.5, 1.5, 1.5, 1.5] (mean
    1.41667), B's = [2, 1.5, 1.5, 1.5, 1.5, 1.5] (mean 1.58333).
    Normalized (num_models=2): A=(1.41667-1)/1=0.41667,
    B=(1.58333-1)/1=0.58333.
    """
    tied = {"core_dice": 0.8, "core_hd95": 3.0, "enhancing_dice": 0.7, "enhancing_hd95": 4.0, "whole_hd95": 2.0}
    df = pd.DataFrame(
        [
            {"model": "A", "case_id": "c1", "whole_dice": 0.9, **tied},
            {"model": "B", "case_id": "c1", "whole_dice": 0.7, **tied},
            {"model": "A", "case_id": "c2", "whole_dice": 0.9, **tied},
            {"model": "B", "case_id": "c2", "whole_dice": 0.7, **tied},
        ]
    )
    scores = compute_brats_ranking(df)
    assert scores["A"] < scores["B"]
    assert abs(scores["A"] - 0.41667) < 1e-4
    assert abs(scores["B"] - 0.58333) < 1e-4


def test_brats_ranking_single_model_does_not_divide_by_zero():
    df = pd.DataFrame(
        [
            {
                "model": "A",
                "case_id": "c1",
                "whole_dice": 0.9,
                "whole_hd95": 2.0,
                "core_dice": 0.8,
                "core_hd95": 3.0,
                "enhancing_dice": 0.7,
                "enhancing_hd95": 4.0,
            }
        ]
    )
    scores = compute_brats_ranking(df)
    assert scores["A"] == 0.0
