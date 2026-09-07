from src.datasets.prepare import assign_folds, assign_holdout_split


def test_fold_assignment_is_deterministic():
    """Same case_ids + same fold_seed must produce the identical fold
    assignment every time — the whole point of writing folds into the
    manifest once (Phase 8) rather than recomputing them per training
    run is to guarantee this. A seed/library-version drift here would
    silently produce 5 non-partitioning splits across separate `train`
    invocations."""
    case_ids = [f"case_{i:03d}" for i in range(20)]
    first = assign_folds(case_ids, num_folds=5, fold_seed=42)
    second = assign_folds(case_ids, num_folds=5, fold_seed=42)
    assert first == second


def test_fold_assignment_is_a_disjoint_partition():
    case_ids = [f"case_{i:03d}" for i in range(20)]
    folds = assign_folds(case_ids, num_folds=5, fold_seed=42)
    assert set(folds.keys()) == set(case_ids)
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_different_seed_gives_different_partition():
    case_ids = [f"case_{i:03d}" for i in range(20)]
    a = assign_folds(case_ids, num_folds=5, fold_seed=42)
    b = assign_folds(case_ids, num_folds=5, fold_seed=1)
    assert a != b


def test_holdout_split_is_disjoint_from_train_pool():
    """Regression test for the Phase 9 gap: ensembling (Table 2) needs
    cases NONE of the 5 fold models trained on — matching the paper's
    own practice. The original Phase 8 implementation only k-fold-split
    the entire dataset, so every case belonged to some fold's own
    validation set, meaning an "ensemble" evaluated on those cases would
    have had 4 of 5 models already trained on each one."""
    case_ids = [f"case_{i:03d}" for i in range(200)]
    splits = assign_holdout_split(case_ids, holdout_ratio=0.1, fold_seed=42)
    holdout = {c for c, s in splits.items() if s == "holdout"}
    train_pool = {c for c, s in splits.items() if s == "train_pool"}
    assert holdout | train_pool == set(case_ids)
    assert holdout & train_pool == set()
    assert 15 <= len(holdout) <= 25  # ~10% of 200, allowing for rounding


def test_folds_only_assigned_within_train_pool():
    """Folds must never include a holdout case — assign_folds() should
    only ever be called on the train_pool subset, and this test guards
    against a future caller accidentally passing the full case list."""
    case_ids = [f"case_{i:03d}" for i in range(200)]
    splits = assign_holdout_split(case_ids, holdout_ratio=0.1, fold_seed=42)
    train_pool = [c for c in case_ids if splits[c] == "train_pool"]
    folds = assign_folds(train_pool, num_folds=5, fold_seed=42)
    assert set(folds.keys()) == set(train_pool)
