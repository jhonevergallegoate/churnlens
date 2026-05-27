"""Tests para `churnlens.features.selection`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnlens.features.selection import (
    chi2_scores,
    consensus_top_k,
    l1_logistic_importance,
    load_training_matrix,
    mutual_information_scores,
    permutation_importance_rf,
    persist_feature_selection,
    run_feature_selection,
)


@pytest.fixture
def toy_xy() -> tuple[pd.DataFrame, pd.Series]:
    """8 features sintéticas + target binario donde solo 3 son informativas."""
    rng = np.random.default_rng(0)
    n = 400
    informative = rng.normal(size=(n, 3))
    noise = rng.normal(size=(n, 5))
    logits = informative @ np.array([1.5, -1.2, 0.8])
    y = pd.Series((logits + rng.normal(scale=0.5, size=n) > 0).astype("int8"), name="y")
    informative_nn = informative - informative.min(axis=0)
    noise_nn = noise - noise.min(axis=0)
    x = pd.DataFrame(
        np.hstack([informative_nn, noise_nn]),
        columns=[f"info_{i}" for i in range(3)] + [f"noise_{i}" for i in range(5)],
    ).astype("float32")
    return x, y


def test_mutual_information_returns_one_score_per_feature(
    toy_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = toy_xy
    scores = mutual_information_scores(x, y, random_state=0)
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == set(x.columns)
    assert (scores >= 0).all()


def test_chi2_skips_negative_columns(toy_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = toy_xy
    x_with_neg = x.copy()
    x_with_neg["neg_col"] = -1.0
    scores = chi2_scores(x_with_neg, y)
    assert pd.isna(scores.loc["neg_col"])
    assert scores.drop("neg_col").notna().all()


def test_l1_logistic_importance_runs(toy_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = toy_xy
    scores = l1_logistic_importance(x, y, random_state=0)
    assert (scores >= 0).all()
    assert scores.index.tolist() == sorted(scores.index, key=lambda c: -scores.loc[c])


def test_permutation_importance_rf_assigns_higher_scores_to_informative(
    toy_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = toy_xy
    scores = permutation_importance_rf(x, y, n_estimators=80, n_repeats=3, random_state=0, n_jobs=1)
    informative = scores.filter(like="info_").mean()
    noise = scores.filter(like="noise_").mean()
    assert informative > noise


def test_consensus_top_k_votes_match_intersection() -> None:
    features = [f"f{i}" for i in range(6)]
    s1 = pd.Series([5, 4, 3, 2, 1, 0], index=features)
    s2 = pd.Series([3, 2, 5, 4, 1, 0], index=features)
    scores_wide, ranks_wide, _consensus, top = consensus_top_k(
        {"a": s1, "b": s2}, k=3, feature_order=features
    )
    assert scores_wide.shape == (6, 2)
    assert ranks_wide.loc["f0", "a"] == 1
    top_set = set(top)
    expected_overlap = {"f0", "f1", "f2", "f3"}
    assert top_set.issubset(expected_overlap)
    assert len(top) == 3


def test_consensus_top_k_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="vacío"):
        consensus_top_k({}, k=3)


def _write_synthetic_train(tmp_path: Path, *, n: int = 200, n_features: int = 8) -> Path:
    rng = np.random.default_rng(0)
    informative = rng.normal(size=(n, 3))
    logits = informative @ np.array([1.5, -1.2, 0.8])
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype("int8")
    noise = rng.normal(size=(n, n_features - 3))
    x = np.hstack([informative, noise])
    x_nn = x - x.min(axis=0)
    cols = [f"info_{i}" for i in range(3)] + [f"noise_{i}" for i in range(n_features - 3)]
    df = pd.DataFrame(x_nn, columns=cols).astype("float32")
    df["Churn"] = y
    path = tmp_path / "train.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def test_load_training_matrix_raises_without_target(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    p = tmp_path / "train.parquet"
    df.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="objetivo"):
        load_training_matrix(p)


def test_run_feature_selection_endtoend_with_synthetic(tmp_path: Path) -> None:
    train_path = _write_synthetic_train(tmp_path)
    result = run_feature_selection(
        train_path=train_path,
        k=4,
        random_state=0,
        rf_estimators=50,
        permutation_repeats=2,
        permutation_n_jobs=1,
    )
    assert result.k == 4
    assert len(result.top_k) == 4
    assert set(result.scores.columns) == {"mutual_info", "chi2", "l1_logreg", "permutation_rf"}
    assert "info_0" in result.top_k or "info_1" in result.top_k or "info_2" in result.top_k


def test_persist_feature_selection_writes_all_artifacts(tmp_path: Path) -> None:
    train_path = _write_synthetic_train(tmp_path)
    result = run_feature_selection(
        train_path=train_path,
        k=4,
        random_state=0,
        rf_estimators=50,
        permutation_repeats=2,
        permutation_n_jobs=1,
    )
    tables_dir = tmp_path / "tables"
    processed_dir = tmp_path / "processed"
    paths = persist_feature_selection(result, tables_dir=tables_dir, processed_dir=processed_dir)
    for key in ("scores", "ranks", "consensus", "manifest"):
        assert paths[key].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["k"] == 4
    assert len(manifest["top_k_features"]) == 4
