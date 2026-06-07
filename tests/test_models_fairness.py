"""Tests para `churnlens.models.fairness`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnlens.config import Settings
from churnlens.models.fairness import (
    SENSITIVE_ATTRIBUTES,
    audit_attributes,
    expected_calibration_error,
    fairness_summary,
    group_metrics,
    plot_fairness_groups,
    run_fairness_audit,
)


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------
def test_ece_perfectly_calibrated_is_zero() -> None:
    # 30 % de positivos con probabilidad constante 0.3 → calibración exacta.
    y = np.array([1] * 30 + [0] * 70, dtype="int8")
    proba = np.full(100, 0.3)
    assert expected_calibration_error(y, proba) == pytest.approx(0.0, abs=1e-12)


def test_ece_overconfident_model_is_penalized() -> None:
    # Probabilidad constante 0.9 con solo 50 % de positivos → ECE = 0.4.
    y = np.array([1] * 50 + [0] * 50, dtype="int8")
    proba = np.full(100, 0.9)
    assert expected_calibration_error(y, proba) == pytest.approx(0.4, abs=1e-12)


def test_ece_handles_extreme_probabilities() -> None:
    # p == 0.0 y p == 1.0 deben caer en los bins extremos sin errores.
    y = np.array([0, 1], dtype="int8")
    proba = np.array([0.0, 1.0])
    assert expected_calibration_error(y, proba) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# group_metrics
# ---------------------------------------------------------------------------
def test_group_metrics_exact_values() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype="int8")
    proba = np.array([0.9, 0.4, 0.8, 0.1, 0.95, 0.2, 0.3, 0.4])
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    table = group_metrics(y, proba, groups, threshold=0.5)

    # Grupo A: y_hat = [1, 0, 1, 0] → selección 0.5, TPR 0.5, FPR 0.5.
    assert table.loc["A", "n"] == 4
    assert table.loc["A", "prevalence"] == pytest.approx(0.5)
    assert table.loc["A", "selection_rate"] == pytest.approx(0.5)
    assert table.loc["A", "tpr"] == pytest.approx(0.5)
    assert table.loc["A", "fpr"] == pytest.approx(0.5)
    assert table.loc["A", "precision"] == pytest.approx(0.5)
    # Grupo B: y_hat = [1, 0, 0, 0] → selección 0.25, TPR 1.0, FPR 0.0.
    assert table.loc["B", "selection_rate"] == pytest.approx(0.25)
    assert table.loc["B", "tpr"] == pytest.approx(1.0)
    assert table.loc["B", "fpr"] == pytest.approx(0.0)
    assert table.loc["B", "precision"] == pytest.approx(1.0)


def test_group_metrics_degenerate_groups_yield_nan() -> None:
    # Grupo A sin positivos → TPR NaN; grupo B sin predicciones positivas
    # → precision NaN.
    y = np.array([0, 0, 1, 0], dtype="int8")
    proba = np.array([0.9, 0.1, 0.2, 0.3])
    groups = pd.Series(["A", "A", "B", "B"])

    table = group_metrics(y, proba, groups, threshold=0.5)

    assert np.isnan(table.loc["A", "tpr"])
    assert np.isnan(table.loc["B", "precision"])
    assert table.loc["B", "selection_rate"] == pytest.approx(0.0)


def test_group_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="misma longitud"):
        group_metrics(
            np.array([0, 1]),
            np.array([0.5, 0.6, 0.7]),
            pd.Series(["A", "B"]),
            threshold=0.5,
        )


# ---------------------------------------------------------------------------
# fairness_summary
# ---------------------------------------------------------------------------
def test_fairness_summary_parity_case_passes_all() -> None:
    # Ambos grupos idénticos y perfectamente calibrados → todo dentro de umbral.
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype="int8")
    proba = np.full(8, 0.5)
    groups = pd.Series(["A"] * 4 + ["B"] * 4)

    summary = fairness_summary(group_metrics(y, proba, groups, threshold=0.5))

    assert summary["disparate_impact"] == pytest.approx(1.0)
    assert summary["demographic_parity_diff"] == pytest.approx(0.0)
    assert summary["equalized_odds_diff"] == pytest.approx(0.0)
    assert summary["max_ece"] == pytest.approx(0.0)
    assert summary["within_thresholds"] is True


def test_fairness_summary_flags_disparity() -> None:
    # Grupo B casi nunca seleccionado → DI muy por debajo de 0.80.
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype="int8")
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.1, 0.2])
    groups = pd.Series(["A"] * 4 + ["B"] * 4)

    summary = fairness_summary(group_metrics(y, proba, groups, threshold=0.5))

    assert summary["disparate_impact"] == pytest.approx(0.0)
    assert summary["disparate_impact_ok"] is False
    assert summary["demographic_parity_diff"] == pytest.approx(1.0)
    assert summary["within_thresholds"] is False


def test_fairness_summary_ignores_nan_gaps() -> None:
    # TPR NaN en un grupo: el EOD cae al gap disponible (FPR).
    table = pd.DataFrame(
        {
            "n": [10, 10],
            "prevalence": [0.0, 0.5],
            "selection_rate": [0.20, 0.22],
            "tpr": [float("nan"), 0.8],
            "fpr": [0.20, 0.25],
            "precision": [float("nan"), 0.6],
            "ece": [0.01, 0.02],
        },
        index=pd.Index(["A", "B"], name="group"),
    )

    summary = fairness_summary(table)

    assert summary["equalized_odds_diff"] == pytest.approx(0.05)
    assert summary["equalized_odds_ok"] is True


# ---------------------------------------------------------------------------
# audit_attributes
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_audit_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 300
    sensitive = pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], size=n),
            "SeniorCitizen": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
        }
    )
    y = rng.choice([0, 1], size=n, p=[0.73, 0.27]).astype("int8")
    proba = np.clip(0.25 + 0.4 * y + rng.normal(scale=0.2, size=n), 0.001, 0.999)
    return sensitive, y, proba


def test_audit_attributes_builds_long_table_and_summary(
    synthetic_audit_inputs: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    sensitive, y, proba = synthetic_audit_inputs
    attributes = ("gender", "SeniorCitizen")

    table, summary = audit_attributes(sensitive, y, proba, threshold=0.5, attributes=attributes)

    assert list(table.index.names) == ["attribute", "group"]
    assert set(table.index.get_level_values("attribute")) == set(attributes)
    assert set(summary) == set(attributes)
    for entry in summary.values():
        assert {"disparate_impact", "demographic_parity_diff", "equalized_odds_diff"} <= set(entry)
        assert isinstance(entry["within_thresholds"], bool)


def test_audit_attributes_missing_column_raises(
    synthetic_audit_inputs: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    sensitive, y, proba = synthetic_audit_inputs
    with pytest.raises(ValueError, match="ausentes"):
        audit_attributes(sensitive, y, proba, threshold=0.5, attributes=("gender", "Partner"))


def test_plot_fairness_groups_writes_png(
    tmp_path: Path,
    synthetic_audit_inputs: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    sensitive, y, proba = synthetic_audit_inputs
    table, _ = audit_attributes(
        sensitive, y, proba, threshold=0.5, attributes=("gender", "SeniorCitizen")
    )

    out = plot_fairness_groups(table, out_path=tmp_path / "fairness.png", threshold=0.5)

    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# run_fairness_audit (integración end-to-end con artefactos sintéticos)
# ---------------------------------------------------------------------------
def test_run_fairness_audit_end_to_end(
    tmp_path: Path, synthetic_churn_dataset: pd.DataFrame
) -> None:
    from sklearn.linear_model import LogisticRegression

    from churnlens.features.pipeline import run_preprocessing
    from churnlens.features.preprocessing import TARGET_COL
    from churnlens.models.registry import save_model

    # 1. Materializa el pipeline de la Fase 2 sobre datos sintéticos.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "telco_customer_churn.csv").write_text(
        synthetic_churn_dataset.to_csv(index=False), encoding="utf-8"
    )
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        log_level="WARNING",
    )
    run_preprocessing(settings=settings, include_engineered=True)

    # 2. Entrena y registra un modelo mínimo compatible con el manifest real.
    train_df = pd.read_parquet(settings.processed_dir / "train.parquet")
    feature_set = [c for c in train_df.columns if c != TARGET_COL]
    model = LogisticRegression(max_iter=500, random_state=42).fit(
        train_df[feature_set].to_numpy(), train_df[TARGET_COL].to_numpy()
    )
    save_model(
        model,
        "logreg_fair_test",
        metadata={
            "feature_set": feature_set,
            "metrics": {"val_tuned": {"threshold": 0.5}},
        },
        models_dir=settings.models_dir,
    )

    # 3. La auditoría reconstruye los atributos sensibles y persiste todo.
    result = run_fairness_audit(model_name="logreg_fair_test", settings=settings)

    assert result.threshold == pytest.approx(0.5)
    assert set(result.summary) == set(SENSITIVE_ATTRIBUTES)
    assert result.n_rows == len(pd.read_parquet(settings.processed_dir / "test.parquet"))
    for path in (result.groups_path, result.summary_path, result.figure_path):
        assert path.exists(), f"Falta artefacto: {path}"
        assert path.stat().st_size > 0

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["model"] == "logreg_fair_test"
    assert payload["split"] == "test"
    assert set(payload["attributes"]) == set(SENSITIVE_ATTRIBUTES)


def test_run_fairness_audit_missing_model_raises(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        log_level="WARNING",
    )
    with pytest.raises(FileNotFoundError):
        run_fairness_audit(model_name="no_existe", settings=settings)
