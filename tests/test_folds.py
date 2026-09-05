from src.datasets.prepare import assign_folds


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
