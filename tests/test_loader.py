"""Tests para `churnlens.data.loader`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churnlens.config import Settings
from churnlens.data.loader import TelcoChurnLoader


@pytest.fixture
def offline_loader(tmp_path: Path, sample_csv_text: str) -> TelcoChurnLoader:
    """Devuelve un loader con un raw CSV pre-existente (sin red)."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_data_dirs()
    settings.raw_csv_path.write_text(sample_csv_text, encoding="utf-8")
    return TelcoChurnLoader(settings=settings)


def test_load_raw_returns_dataframe(offline_loader: TelcoChurnLoader) -> None:
    df = offline_loader.load_raw()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 21
    assert len(df) >= 1


def test_load_raw_raises_when_file_missing(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_data_dirs()
    loader = TelcoChurnLoader(settings=settings)
    with pytest.raises(FileNotFoundError):
        loader.load_raw()


def test_coerce_types_handles_empty_total_charges(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Las cadenas vacías de TotalCharges deben volverse NaN float32."""
    out = TelcoChurnLoader._coerce_types(sample_dataframe)
    assert out["TotalCharges"].dtype == "float32"
    assert out["TotalCharges"].isna().sum() >= 1


def test_coerce_types_sets_contract_ordered(
    sample_dataframe: pd.DataFrame,
) -> None:
    out = TelcoChurnLoader._coerce_types(sample_dataframe)
    assert isinstance(out["Contract"].dtype, pd.CategoricalDtype)
    assert out["Contract"].cat.ordered is True


def test_load_validated_passes_for_sample(offline_loader: TelcoChurnLoader) -> None:
    df = offline_loader.load_validated()
    assert "Churn" in df.columns
    assert df["Churn"].dtype.name == "category"


def test_materialize_interim_writes_parquet(offline_loader: TelcoChurnLoader) -> None:
    parquet_path = offline_loader.materialize_interim()
    assert parquet_path.exists()
    assert parquet_path.suffix == ".parquet"
    df = pd.read_parquet(parquet_path)
    assert df.shape[1] == 21


def test_summary_returns_consistent_metrics(
    offline_loader: TelcoChurnLoader,
) -> None:
    # Forzamos el cálculo y persistencia de checksums.
    offline_loader._record_checksums(offline_loader.settings.raw_csv_path)
    summary = offline_loader.summary()
    d = summary.to_dict()
    assert d["n_rows"] >= 1
    assert d["n_cols"] == 21
    assert 0.0 <= float(d["target_pos_rate"]) <= 1.0
    assert len(str(d["md5"])) == 32
