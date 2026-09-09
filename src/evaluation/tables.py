"""Formats the raw per-case eval_results_*.json files into the paper's
actual Table 1 and Table 2 layouts (mean Dice/HD95 per region, per
variant) — the piece that was still missing after Phase 9's evaluation
step: ranking.py (Table 3) already existed, but nothing turned the raw
per-case JSON into a human-readable table matching the paper's own
column layout.

Table 1 (paper Section 3.1, "Training set (Dice, 5-fold CV)"):
    one row per variant, columns Whole/Core/Enh/Mean, Dice only.
    Built from results/tables/cv_val/eval_results_<variant>_fold<i>.json
    — pooled across all 5 folds (each case validated exactly once by
    its own fold, so pooling all folds' per-case results together and
    averaging is equivalent to the paper's "provides a performance
    estimate on the training cases" description, Section 3.1).

Table 2 (paper Section 3.1, "Validation set"):
    one row per variant, Dice columns (Whole/Core/Enh/Mean) AND HD95
    columns (Whole/Core/Enh/Mean). Built from the ensembled
    results/tables/eval_results_<variant>.json (the holdout split) —
    the closest equivalent this project has to the paper's real BraTS
    validation set (see Phase 8's own stated limitation: no access to
    BraTS's actual online judge, so this is a local held-out stand-in).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REGIONS = ["whole", "core", "enhancing"]


def _load_cases(paths: list[Path]) -> list[dict]:
    cases = []
    for path in paths:
        with open(path) as f:
            cases.extend(json.load(f))
    return cases


def build_table1(cv_val_dir: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        fold_files = sorted(cv_val_dir.glob(f"eval_results_{variant}_fold*.json"))
        if not fold_files:
            continue
        cases = _load_cases(fold_files)
        df = pd.DataFrame(cases)
        row: dict[str, object] = {"Model": variant}
        dice_cols = [f"{r}_dice" for r in REGIONS]
        for region, col in zip(["Whole", "Core", "Enh"], dice_cols):
            row[region] = round(df[col].mean() * 100, 2)
        row["Mean"] = round(df[dice_cols].mean(axis=1).mean() * 100, 2)
        row["n_cases"] = len(df)
        rows.append(row)
    return pd.DataFrame(rows)


def build_table2(holdout_dir: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        path = holdout_dir / f"eval_results_{variant}.json"
        if not path.exists():
            continue
        cases = _load_cases([path])
        df = pd.DataFrame(cases)
        dice_cols = [f"{r}_dice" for r in REGIONS]
        hd95_cols = [f"{r}_hd95" for r in REGIONS]
        row: dict[str, object] = {"Model": variant}
        for region, col in zip(["Whole", "Core", "Enh"], dice_cols):
            row[f"Dice_{region}"] = round(df[col].mean() * 100, 2)
        row["Dice_Mean"] = round(df[dice_cols].mean(axis=1).mean() * 100, 2)
        for region, col in zip(["Whole", "Core", "Enh"], hd95_cols):
            row[f"HD95_{region}"] = round(df[col].mean(), 3)
        row["HD95_Mean"] = round(df[hd95_cols].mean(axis=1).mean(), 3)
        row["n_cases"] = len(df)
        rows.append(row)
    return pd.DataFrame(rows)


VARIANTS = [
    "baseline",
    "baseline_bs5",
    "ablation_region",
    "ablation_dataug",
    "ablation_dataug_star",
    "ablation_batchnorm",
    "ablation_batchdice",
    "ablation_dataug_star_bn",
]


def _toy_example() -> None:
    """Hand-verifiable self-test, same pattern as ranking.py's own toy
    example: 2 cases for one variant, known Dice values, confirms the
    mean computation is exactly what it should be."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cv_val_dir = Path(tmp) / "cv_val"
        cv_val_dir.mkdir()
        cases = [
            {
                "case_id": "c1",
                "whole_dice": 0.9,
                "core_dice": 0.8,
                "enhancing_dice": 0.7,
                "whole_hd95": 2.0,
                "core_hd95": 3.0,
                "enhancing_hd95": 4.0,
            },
            {
                "case_id": "c2",
                "whole_dice": 0.8,
                "core_dice": 0.6,
                "enhancing_dice": 0.5,
                "whole_hd95": 2.0,
                "core_hd95": 3.0,
                "enhancing_hd95": 4.0,
            },
        ]
        with open(cv_val_dir / "eval_results_toy_fold0.json", "w") as f:
            json.dump(cases, f)
        table1 = build_table1(cv_val_dir, ["toy"])
        print(table1)
        # by hand: Whole=(0.9+0.8)/2*100=85.0, Core=(0.8+0.6)/2*100=70.0, Enh=(0.7+0.5)/2*100=60.0
        # Mean = mean of [85,70,60] per-case-then-overall = ((0.9+0.8+0.7)/3 + (0.8+0.6+0.5)/3)/2*100 = 71.67
        assert table1.iloc[0]["Whole"] == 85.0
        assert table1.iloc[0]["Core"] == 70.0
        assert table1.iloc[0]["Enh"] == 60.0
        assert abs(table1.iloc[0]["Mean"] - 71.67) < 0.01
        print("PASS: Table 1 toy example matches hand-computed values exactly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/tables")
    args, _ = parser.parse_known_args()

    results_dir = Path(args.results_dir)
    cv_val_dir = results_dir / "cv_val"

    if not cv_val_dir.exists() or not any(cv_val_dir.glob("eval_results_*.json")):
        print(f"No cv_val results found under {cv_val_dir} — running the toy-example self-test instead.")
        _toy_example()
    else:
        table1 = build_table1(cv_val_dir, VARIANTS)
        table2 = build_table2(results_dir, VARIANTS)

        print("=== Table 1: Training set (Dice, 5-fold CV) ===")
        print(table1.to_string(index=False))
        print()
        print("=== Table 2: Validation set (ensembled, local holdout) ===")
        print(table2.to_string(index=False))

        table1.to_csv(results_dir / "table1.csv", index=False)
        table2.to_csv(results_dir / "table2.csv", index=False)
        print(f"\nWrote {results_dir / 'table1.csv'} and {results_dir / 'table2.csv'}")