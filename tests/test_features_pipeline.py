"""Tests para `churnlens.features.pipeline`."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from churnlens.config import Settings
from churnlens.features.pipeline import run_preprocessing


def _seed_raw_dataset(tmp_data_dir: Path, raw_csv_text: str) -> None:
    raw_dir = tmp_data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "telco_customer_churn.csv").write_text(raw_csv_text, encoding="utf-8")


def test_run_preprocessing_writes_all_artifacts(
    tmp_path: Path, synthetic_churn_dataset: pd.DataFrame
) -> None:
    # Reusa el dataset sintético escribiéndolo como CSV en una raíz temporal.
    csv_text = synthetic_churn_dataset.to_csv(index=False)
    _seed_raw_dataset(tmp_path, csv_text)

    settings = Settings(
        data_dir=tmp_path,
        log_level="WARNING",
    )

    artifacts = run_preprocessing(settings=settings, include_engineered=True)

    for path in (
        artifacts.train_path,
        artifacts.val_path,
        artifacts.test_path,
        artifacts.preprocessor_path,
        artifacts.feature_names_path,
        artifacts.metadata_path,
    ):
        assert path.exists(), f"Falta artefacto: {path}"
        assert path.stat().st_size > 0

    # El preprocesador debe ser deserializable y los nombres deben coincidir.
    joblib.load(artifacts.preprocessor_path)
    feature_names = json.loads(artifacts.feature_names_path.read_text("utf-8"))
    assert len(feature_names) > 0

    # Los parquet deben tener tantas columnas como features + target.
    train_df = pd.read_parquet(artifacts.train_path)
    assert "Churn" in train_df.columns
    assert train_df.shape[1] == len(feature_names) + 1

    # Las tasas de positivos deben ser similares entre splits.
    metadata = json.loads(artifacts.metadata_path.read_text("utf-8"))
    rates = [s["positive_rate"] for s in metadata["splits"].values()]
    assert max(rates) - min(rates) < 0.1


def test_run_preprocessing_supports_no_engineered_flag(
    tmp_path: Path, synthetic_churn_dataset: pd.DataFrame
) -> None:
    csv_text = synthetic_churn_dataset.to_csv(index=False)
    _seed_raw_dataset(tmp_path, csv_text)

    settings = Settings(data_dir=tmp_path, log_level="WARNING")
    artifacts_with = run_preprocessing(settings=settings, include_engineered=True)
    # Snapshot del primer run antes de que el segundo sobrescriba los artefactos.
    with_names = json.loads(artifacts_with.feature_names_path.read_text("utf-8"))

    artifacts_without = run_preprocessing(settings=settings, include_engineered=False)
    without_names = json.loads(artifacts_without.feature_names_path.read_text("utf-8"))

    assert len(with_names) > len(without_names)
