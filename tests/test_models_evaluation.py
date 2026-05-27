"""Tests para `churnlens.models.evaluation`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnlens.models.evaluation import (
    DEFAULT_THRESHOLDS,
    binary_metrics,
    optimal_threshold,
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sweep,
    save_metrics_json,
    threshold_sweep,
)


@pytest.fixture
def y_and_proba() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 400
    y = rng.choice([0, 1], size=n, p=[0.7, 0.3]).astype("int8")
    noise = rng.normal(scale=0.4, size=n)
    proba = np.clip(0.2 + 0.5 * y + noise, 0.001, 0.999)
    return y, proba


def test_binary_metrics_returns_expected_keys(y_and_proba: tuple[np.ndarray, np.ndarray]) -> None:
    y, proba = y_and_proba
    metrics = binary_metrics(y, proba, threshold=0.5)
    expected = {
        "pr_auc",
        "roc_auc",
        "brier",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "positive_rate",
        "base_rate",
        "lift_at_10",
        "threshold",
    }
    assert expected.issubset(metrics)
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["lift_at_10"] >= 0


def test_threshold_sweep_is_indexed_by_threshold(
    y_and_proba: tuple[np.ndarray, np.ndarray],
) -> None:
    y, proba = y_and_proba
    df = threshold_sweep(y, proba, thresholds=(0.1, 0.3, 0.5, 0.7))
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == [0.1, 0.3, 0.5, 0.7]
    assert {"precision", "recall", "f1", "accuracy", "positive_rate"}.issubset(df.columns)


def test_optimal_threshold_picks_within_grid(y_and_proba: tuple[np.ndarray, np.ndarray]) -> None:
    y, proba = y_and_proba
    choice = optimal_threshold(y, proba, metric="f1")
    assert choice.metric == "f1"
    assert choice.threshold in {round(t, 4) for t in DEFAULT_THRESHOLDS}
    assert 0.0 <= choice.score <= 1.0


def test_optimal_threshold_rejects_unknown_metric(
    y_and_proba: tuple[np.ndarray, np.ndarray],
) -> None:
    y, proba = y_and_proba
    with pytest.raises(ValueError, match="no soportada"):
        optimal_threshold(y, proba, metric="ndcg")


def test_plot_pr_curves_writes_png(
    tmp_path: Path, y_and_proba: tuple[np.ndarray, np.ndarray]
) -> None:
    y, proba = y_and_proba
    out = plot_pr_curves({"toy": (y, proba)}, out_path=tmp_path / "pr.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_roc_curves_writes_png(
    tmp_path: Path, y_and_proba: tuple[np.ndarray, np.ndarray]
) -> None:
    y, proba = y_and_proba
    out = plot_roc_curves({"toy": (y, proba)}, out_path=tmp_path / "roc.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_writes_png(
    tmp_path: Path, y_and_proba: tuple[np.ndarray, np.ndarray]
) -> None:
    y, proba = y_and_proba
    out = plot_confusion_matrix(y, (proba >= 0.5).astype("int8"), out_path=tmp_path / "cm.png")
    assert out.exists()


def test_plot_calibration_writes_png(
    tmp_path: Path, y_and_proba: tuple[np.ndarray, np.ndarray]
) -> None:
    y, proba = y_and_proba
    out = plot_calibration(y, proba, out_path=tmp_path / "cal.png")
    assert out.exists()


def test_plot_threshold_sweep_writes_png(
    tmp_path: Path, y_and_proba: tuple[np.ndarray, np.ndarray]
) -> None:
    y, proba = y_and_proba
    sweep = threshold_sweep(y, proba, thresholds=(0.2, 0.4, 0.6, 0.8))
    out = plot_threshold_sweep(sweep, out_path=tmp_path / "thr.png", chosen=0.4)
    assert out.exists()


def test_plot_feature_importance_writes_png(tmp_path: Path) -> None:
    importance = pd.Series([0.1, 0.4, 0.2], index=["a", "b", "c"])
    out = plot_feature_importance(importance, out_path=tmp_path / "imp.png", top_n=3)
    assert out.exists()


def test_save_metrics_json_roundtrip(tmp_path: Path) -> None:
    import json

    p = save_metrics_json({"a": 1.0, "b": 0.5}, tmp_path / "metrics.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload == {"a": 1.0, "b": 0.5}
