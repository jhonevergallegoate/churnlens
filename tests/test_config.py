"""Tests para `churnlens.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from churnlens.config import Settings


def test_settings_defaults() -> None:
    """Los defaults críticos deben corresponder a los documentados."""
    s = Settings()
    assert s.log_level == "INFO"
    assert s.log_format == "console"
    assert s.raw_filename == "telco_customer_churn.csv"
    assert s.random_seed == 42
    assert s.data_url.startswith("https://")


def test_settings_paths_are_relative_to_project_root() -> None:
    """Las rutas derivadas deben colgar de `project_root`."""
    s = Settings()
    assert s.raw_dir == s.data_dir / "raw"
    assert s.interim_dir == s.data_dir / "interim"
    assert s.processed_dir == s.data_dir / "processed"
    assert s.raw_csv_path.name == "telco_customer_churn.csv"
    assert s.interim_parquet_path.suffix == ".parquet"


def test_log_level_validation() -> None:
    """Niveles inválidos deben lanzar `ValueError`."""
    with pytest.raises(ValueError, match="log_level"):
        Settings(log_level="VERBOSE")


def test_log_format_validation() -> None:
    with pytest.raises(ValueError, match="log_format"):
        Settings(log_format="xml")


def test_ensure_data_dirs_creates_directories(tmp_path: Path) -> None:
    """`ensure_data_dirs` debe crear las tres carpetas estándar."""
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_data_dirs()
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "interim").is_dir()
    assert (tmp_path / "data" / "processed").is_dir()
