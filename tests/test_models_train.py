"""Tests para `churnlens.models.train` (end-to-end con datos sintéticos)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnlens.models.train import (
    MODEL_SPECS,
    quick_baseline_only,
    train_models,
)


@pytest.fixture
def synthetic_processed(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Genera train/val parquet sintéticos con 5 features informativas."""
    rng = np.random.default_rng(0)

    def make(n: int) -> pd.DataFrame:
        x_inf = rng.normal(size=(n, 3))
        logits = x_inf @ np.array([1.6, -1.4, 1.1])
        y = (logits + rng.normal(scale=0.6, size=n) > 0).astype("int8")
        x_noise = rng.normal(size=(n, 2))
        cols = [f"info_{i}" for i in range(3)] + [f"noise_{i}" for i in range(2)]
        df = pd.DataFrame(np.hstack([x_inf, x_noise]), columns=cols).astype("float32")
        df["Churn"] = y
        return df

    train_df = make(400)
    val_df = make(100)
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    train_df.to_parquet(train_p, index=False)
    val_df.to_parquet(val_p, index=False)
    models_dir = tmp_path / "models"
    tables_dir = tmp_path / "tables"
    figs_dir = tmp_path / "figures"
    return train_p, val_p, models_dir, tables_dir, figs_dir


def test_model_specs_catalog_complete() -> None:
    expected = {
        "dummy_stratified",
        "dummy_most_frequent",
        "dummy_prior",
        "logreg_balanced",
        "logreg_l1",
        "random_forest",
        "hist_gb",
        "lightgbm",
    }
    assert expected.issubset(set(MODEL_SPECS))


def test_train_models_baselines_only_end_to_end(
    synthetic_processed: tuple[Path, Path, Path, Path, Path],
) -> None:
    train_p, val_p, models_dir, tables_dir, figs_dir = synthetic_processed
    artifacts = train_models(
        train_path=train_p,
        val_path=val_p,
        models=["dummy_stratified", "logreg_balanced"],
        cv=3,
        random_state=0,
        models_dir=models_dir,
        tables_dir=tables_dir,
        figures_dir=figs_dir,
    )
    assert set(artifacts.model_entries) == {"dummy_stratified", "logreg_balanced"}
    assert artifacts.summary_table.shape[0] == 2
    expected_cols = {
        "model",
        "val_pr_auc",
        "cv_pr_auc_mean",
        "val_roc_auc",
        "val_f1_tuned",
        "val_threshold_tuned",
    }
    assert expected_cols.issubset(set(artifacts.summary_table.columns))
    # logreg debería superar al dummy en PR-AUC sobre datos lineales sintéticos.
    pr = artifacts.summary_table.set_index("model")["val_pr_auc"]
    assert pr["logreg_balanced"] > pr["dummy_stratified"]
    # Artefactos en disco
    assert (tables_dir / "modeling_summary.csv").exists()
    assert (tables_dir / "modeling_cv_scores.csv").exists()
    assert (tables_dir / "modeling_best_summary.json").exists()
    assert (figs_dir / "modeling_pr_curves_val.png").exists()
    assert (figs_dir / "modeling_roc_curves_val.png").exists()
    assert (models_dir / "logreg_balanced.joblib").exists()
    assert (models_dir / "logreg_balanced.metadata.json").exists()


def test_train_models_with_feature_subset(
    synthetic_processed: tuple[Path, Path, Path, Path, Path],
) -> None:
    train_p, val_p, models_dir, tables_dir, figs_dir = synthetic_processed
    subset = ["info_0", "info_1"]
    artifacts = train_models(
        train_path=train_p,
        val_path=val_p,
        models=["logreg_balanced"],
        feature_subset=subset,
        cv=3,
        random_state=0,
        models_dir=models_dir,
        tables_dir=tables_dir,
        figures_dir=figs_dir,
    )
    entry = artifacts.model_entries["logreg_balanced"]
    assert entry.metadata["feature_set"] == subset
    assert entry.metadata["feature_subset_applied"] is True


def test_train_models_rejects_unknown_model(
    synthetic_processed: tuple[Path, Path, Path, Path, Path],
) -> None:
    train_p, val_p, models_dir, tables_dir, figs_dir = synthetic_processed
    with pytest.raises(ValueError, match="desconocidos"):
        train_models(
            train_path=train_p,
            val_path=val_p,
            models=["unknown_model"],
            models_dir=models_dir,
            tables_dir=tables_dir,
            figures_dir=figs_dir,
        )


def test_quick_baseline_only_wrapper(
    synthetic_processed: tuple[Path, Path, Path, Path, Path],
) -> None:
    train_p, val_p, models_dir, tables_dir, figs_dir = synthetic_processed
    artifacts = quick_baseline_only(
        train_path=train_p,
        val_path=val_p,
        random_state=0,
        models_dir=models_dir,
        tables_dir=tables_dir,
        figures_dir=figs_dir,
    )
    assert "logreg_balanced" in artifacts.model_entries
    assert "dummy_prior" in artifacts.model_entries
