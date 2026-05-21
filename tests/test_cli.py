"""Tests de la CLI principal del paquete `churnlens`.

Usa `typer.testing.CliRunner` para invocar los sub-comandos en proceso y
verificar tanto los códigos de salida como el contenido principal.

No depende de red ni de la descarga real del dataset: los tests que
requieren datos crudos los siembran como CSV dentro de un `tmp_path`,
exactamente como hace `test_features_pipeline`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from churnlens.cli import app

runner = CliRunner()


def _seed_raw_dataset(tmp_path: Path, raw_csv_text: str) -> Path:
    """Escribe un CSV crudo bajo `tmp_path/raw/`."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv = raw_dir / "telco_customer_churn.csv"
    csv.write_text(raw_csv_text, encoding="utf-8")
    return csv


@pytest.fixture
def cli_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_churn_dataset: pd.DataFrame,
) -> Path:
    """Aísla la CLI en un `tmp_path` con un CSV sintético ya sembrado.

    Parchea la instancia global ``settings`` en todos los módulos que la
    importaron (cli, loader, eda.report) para que la CLI lea y escriba
    dentro de ``tmp_path``.
    """
    csv_text = synthetic_churn_dataset.to_csv(index=False)
    _seed_raw_dataset(tmp_path, csv_text)

    from churnlens import cli as cli_module
    from churnlens import config as config_module
    from churnlens.config import Settings
    from churnlens.data import loader as loader_module
    from churnlens.eda import report as eda_report_module
    from churnlens.features import pipeline as pipeline_module

    test_settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(config_module, "settings", test_settings)
    monkeypatch.setattr(cli_module, "settings", test_settings)
    monkeypatch.setattr(loader_module, "default_settings", test_settings)
    monkeypatch.setattr(eda_report_module, "default_settings", test_settings)
    monkeypatch.setattr(pipeline_module, "default_settings", test_settings)
    return tmp_path


class TestInfo:
    def test_info_prints_version_and_paths(self) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Versión" in result.stdout
        assert "Random seed" in result.stdout


class TestDataSubcommands:
    def test_validate_with_seeded_csv(self, cli_env: Path) -> None:
        result = runner.invoke(app, ["data", "validate"])
        assert result.exit_code == 0, result.stdout
        assert "Esquema válido" in result.stdout

    def test_materialize_writes_parquet(self, cli_env: Path) -> None:
        result = runner.invoke(app, ["data", "materialize"])
        assert result.exit_code == 0, result.stdout
        assert (cli_env / "interim" / "telco_customer_churn.parquet").exists()

    def test_summary_json_output(self, cli_env: Path) -> None:
        result = runner.invoke(app, ["data", "summary", "--json"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["n_rows"] > 0
        assert 0.0 <= payload["target_pos_rate"] <= 1.0


class TestPreprocessSubcommand:
    def test_preprocess_run_creates_artifacts(self, cli_env: Path) -> None:
        result = runner.invoke(app, ["preprocess", "run"])
        assert result.exit_code == 0, result.stdout
        processed = cli_env / "processed"
        assert (processed / "train.parquet").exists()
        assert (processed / "val.parquet").exists()
        assert (processed / "test.parquet").exists()
        assert (processed / "preprocessor.joblib").exists()
        assert (processed / "feature_names.json").exists()
        assert (processed / "metadata.json").exists()

    def test_preprocess_no_engineered_flag(self, cli_env: Path) -> None:
        result = runner.invoke(app, ["preprocess", "run", "--no-engineered"])
        assert result.exit_code == 0, result.stdout
        feature_names = json.loads(
            (cli_env / "processed" / "feature_names.json").read_text("utf-8")
        )
        # Sin features derivadas debe haber menos columnas de salida.
        assert len(feature_names) < 35


class TestEdaSubcommand:
    def test_eda_report_creates_figures_and_tables(
        self, cli_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Aísla también los directorios de reports.
        figures_dir = tmp_path / "figures"
        tables_dir = tmp_path / "tables"

        from churnlens.eda import report as report_module

        original = report_module.generate_eda_report

        def patched(**kwargs: object) -> object:
            kwargs.setdefault("figures_dir", figures_dir)
            kwargs.setdefault("tables_dir", tables_dir)
            return original(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(report_module, "generate_eda_report", patched)
        # Como el CLI hace `from churnlens.eda.report import generate_eda_report`
        # dentro del comando, basta con parchar el módulo origen.

        result = runner.invoke(app, ["eda", "report"])
        assert result.exit_code == 0, result.stdout
        assert any(figures_dir.glob("eda_*.png"))
        assert any(tables_dir.glob("eda_*.csv"))
