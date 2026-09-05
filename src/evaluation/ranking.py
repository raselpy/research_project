"""Reimplementation of the BraTS "rank then aggregate" scheme (Section
2.1). For each test case, each model gets 6 ranks (3 regions x 2
metrics: Dice, HD95). Ranks are averaged across all cases and metrics
per model, then normalized by the number of participating models into
a ranking score from 0 (best) to 1 (worst).

Empty-reference handling for enhancing tumor (Section 2.1's "essentially
binary" observation) happens upstream, at score-computation time, in
src/evaluation/metrics.py's hausdorff95()/dice_score() — both already
return BraTS's own sentinel values (Dice 1/HD95 0 for a correct empty
prediction, Dice 0/HD95 373.13 for a false positive) rather than NaN or
undefined, which is what makes the rank computation here well-defined
without any special-casing.
"""

import json
from pathlib import Path

import pandas as pd

REGIONS = ["whole", "core", "enhancing"]
METRICS = ["dice", "hd95"]


def compute_case_ranks(per_case_scores: pd.DataFrame) -> pd.DataFrame:
    """per_case_scores: rows = (model, case_id), columns =
    whole_dice, whole_hd95, core_dice, core_hd95, enh_dice, enh_hd95
    (renamed here to whole_dice/whole_hd95/core_dice/core_hd95/
    enhancing_dice/enhancing_hd95 to match src/evaluation/run.py's
    actual column names).

    Returns the same shape with each score column replaced by its rank
    among models, computed independently per case. Higher Dice = better
    (ascending=False); lower HD95 = better (ascending=True). Ties share
    the average rank (pandas' default 'average' method), matching
    BraTS's own shared-rank handling described in Section 2.1.
    """
    out = per_case_scores.copy()
    for region in REGIONS:
        out[f"{region}_dice"] = per_case_scores.groupby("case_id")[f"{region}_dice"].rank(
            ascending=False, method="average"
        )
        out[f"{region}_hd95"] = per_case_scores.groupby("case_id")[f"{region}_hd95"].rank(
            ascending=True, method="average"
        )
    return out


def compute_brats_ranking(per_case_scores: pd.DataFrame) -> pd.Series:
    """Returns one ranking score per model, 0 (best) to 1 (worst)."""
    ranks = compute_case_ranks(per_case_scores)
    rank_cols = [f"{r}_{m}" for r in REGIONS for m in METRICS]
    mean_rank_per_model = ranks.groupby("model")[rank_cols].mean().mean(axis=1)
    num_models = per_case_scores["model"].nunique()
    if num_models == 1:
        # a single model always "wins" trivially — avoid a divide-by-zero
        # on (num_models - 1) rather than let it produce NaN/inf silently.
        return pd.Series(0.0, index=mean_rank_per_model.index)
    return (mean_rank_per_model - 1) / (num_models - 1)  # normalize to [0, 1]


def load_per_case_scores(eval_results_by_model: dict[str, list[dict]]) -> pd.DataFrame:
    """Builds the per_case_scores DataFrame compute_brats_ranking expects
    from {model_name: eval_results.json contents} (src/evaluation/run.py's
    output format, one dict per case with case_id + 6 score columns)."""
    rows = []
    for model_name, case_results in eval_results_by_model.items():
        for case_result in case_results:
            rows.append({"model": model_name, **case_result})
    return pd.DataFrame(rows)


def build_ranking_table(eval_results_dir: Path) -> pd.DataFrame:
    """Reads every results/tables/eval_results_<experiment>.json (one per
    trained experiment variant, see src/evaluation/run.py), computes the
    BraTS ranking across all of them, and returns a table matching the
    paper's Table 3 shape: one row per model, a `value` column (the
    ranking score) and a `rank` column (1 = best)."""
    eval_results_by_model = {}
    for path in sorted(eval_results_dir.glob("eval_results_*.json")):
        model_name = path.stem.removeprefix("eval_results_")
        with open(path) as f:
            eval_results_by_model[model_name] = json.load(f)

    if not eval_results_by_model:
        raise FileNotFoundError(f"No eval_results_*.json files found under {eval_results_dir}")

    per_case_scores = load_per_case_scores(eval_results_by_model)
    ranking_scores = compute_brats_ranking(per_case_scores)

    table = ranking_scores.rename("value").reset_index().rename(columns={"index": "model"})
    table["rank"] = table["value"].rank(ascending=True, method="min").astype(int)
    return table.sort_values("rank").reset_index(drop=True)


def _run_toy_example() -> None:
    """Hand-computable acceptance check (Phase 8, 8.3): 2 models, 2
    cases, only whole_dice varies (others tied) — model A should win.
    By hand: A's ranks per case = [1, 1.5, 1.5, 1.5, 1.5, 1.5] (mean
    1.41667), B's = [2, 1.5, 1.5, 1.5, 1.5, 1.5] (mean 1.58333).
    Normalized (num_models=2): A=(1.41667-1)/1=0.41667,
    B=(1.58333-1)/1=0.58333 — matches the printed output exactly."""
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
            },
            {
                "model": "B",
                "case_id": "c1",
                "whole_dice": 0.7,
                "whole_hd95": 2.0,
                "core_dice": 0.8,
                "core_hd95": 3.0,
                "enhancing_dice": 0.7,
                "enhancing_hd95": 4.0,
            },
            {
                "model": "A",
                "case_id": "c2",
                "whole_dice": 0.9,
                "whole_hd95": 2.0,
                "core_dice": 0.8,
                "core_hd95": 3.0,
                "enhancing_dice": 0.7,
                "enhancing_hd95": 4.0,
            },
            {
                "model": "B",
                "case_id": "c2",
                "whole_dice": 0.7,
                "whole_hd95": 2.0,
                "core_dice": 0.8,
                "core_hd95": 3.0,
                "enhancing_dice": 0.7,
                "enhancing_hd95": 4.0,
            },
        ]
    )
    scores = compute_brats_ranking(df)
    print(scores)
    assert scores["A"] < scores["B"], "model A strictly dominates on whole_dice and should rank better (lower score)"
    print("PASS: model A (strictly better on whole_dice, tied elsewhere) ranks better than model B")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/tables")
    parser.add_argument("--out", type=str, default="results/tables/brats_ranking.json")
    args, _ = parser.parse_known_args()

    results_dir = Path(args.results_dir)
    if not list(results_dir.glob("eval_results_*.json")):
        print(f"No eval_results_*.json found under {results_dir} — running the toy-example self-test instead.")
        _run_toy_example()
    else:
        table = build_ranking_table(results_dir)
        print(table)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_json(out_path, orient="records", indent=2)
        print(f"Wrote ranking table for {len(table)} model(s) to {out_path}")
