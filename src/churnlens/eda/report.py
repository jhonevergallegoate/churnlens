"""Orquestador del reporte EDA reproducible.

`generate_eda_report` produce, dentro de los directorios estándar del
proyecto, los siguientes artefactos:

* ``reports/figures/eda_*.png`` — 8 figuras (target, missing, numéricas
  univariado, numéricas vs target, correlación, churn rate por contrato,
  pago y tenure bucket).
* ``reports/tables/*.csv`` — 4 tablas (resumen numérico, resumen
  categórico, V de Cramér, distribución del target).

El módulo está pensado para ser invocado desde:

* La CLI (`churnlens eda report`).
* El script TDSP (`scripts/eda/main.py`).
* El notebook 02 (a través de la función `generate_eda_report`).

Se garantiza idempotencia: ejecutar dos veces sobre el mismo dataset
produce los mismos archivos byte-a-byte (excepto la metadata interna de
PNG, que matplotlib ya normaliza por defecto).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from churnlens.config import Settings
from churnlens.config import settings as default_settings
from churnlens.data.loader import TelcoChurnLoader
from churnlens.eda.plots import (
    plot_categorical_churn_rate,
    plot_correlation_heatmap,
    plot_missing_bar,
    plot_numeric_boxplot_by_target,
    plot_numeric_histogram,
    plot_target_distribution,
)
from churnlens.eda.summary import (
    categorical_summary,
    cramers_v_vs_target,
    numeric_correlation,
    numeric_summary,
    target_distribution,
)
from churnlens.features.engineering import add_engineered_features
from churnlens.features.preprocessing import (
    BINARY_CATEGORICAL_COLS,
    NOMINAL_CATEGORICAL_COLS,
    ORDINAL_COLS,
    TARGET_COL,
)
from churnlens.logger import get_logger

log = get_logger(__name__)

_NUMERIC_COLS_FOR_EDA = ("tenure", "MonthlyCharges", "TotalCharges")
_FIG_SIZE_SMALL: tuple[float, float] = (6.5, 4.0)
_FIG_SIZE_WIDE: tuple[float, float] = (8.0, 5.0)


@dataclass
class EDAReport:
    """Resultado del reporte EDA: rutas de artefactos generados."""

    figures: dict[str, Path] = field(default_factory=dict)
    tables: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Convierte a dict serializable (rutas como string)."""
        return {
            "figures": {k: str(v) for k, v in self.figures.items()},
            "tables": {k: str(v) for k, v in self.tables.items()},
        }


def generate_eda_report(
    *,
    df: pd.DataFrame | None = None,
    settings: Settings | None = None,
    figures_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> EDAReport:
    """Genera todas las figuras y tablas del reporte EDA.

    Args:
        df: DataFrame opcional ya validado y con features derivadas. Si es
            ``None``, se carga vía `TelcoChurnLoader` y se enriquece con
            `add_engineered_features`.
        settings: configuración del proyecto (rutas, semilla).
        figures_dir: directorio para PNG (default: ``reports/figures``).
        tables_dir: directorio para CSV (default: ``reports/tables``).

    Returns:
        :class:`EDAReport` con la ubicación de cada artefacto generado.
    """
    settings = settings or default_settings
    figures_dir = figures_dir or settings.project_root / "reports" / "figures"
    tables_dir = tables_dir or settings.project_root / "reports" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    matplotlib.use("Agg", force=False)

    if df is None:
        loader = TelcoChurnLoader(settings=settings)
        df = add_engineered_features(loader.load_validated())

    report = EDAReport()

    # ------------------------------------------------------------------
    # Tablas
    # ------------------------------------------------------------------
    report.tables["numeric_summary"] = _dump_csv(
        numeric_summary(df, columns=list(_NUMERIC_COLS_FOR_EDA)),
        tables_dir / "eda_numeric_summary.csv",
    )
    report.tables["categorical_summary"] = _dump_csv(
        categorical_summary(
            df,
            columns=[
                *BINARY_CATEGORICAL_COLS,
                *ORDINAL_COLS,
                *NOMINAL_CATEGORICAL_COLS,
                TARGET_COL,
            ],
        ),
        tables_dir / "eda_categorical_summary.csv",
    )
    report.tables["target_distribution"] = _dump_csv(
        target_distribution(df).reset_index(),
        tables_dir / "eda_target_distribution.csv",
        index=False,
    )
    cramers_cols = [
        *BINARY_CATEGORICAL_COLS,
        *ORDINAL_COLS,
        *NOMINAL_CATEGORICAL_COLS,
    ]
    report.tables["cramers_v"] = _dump_csv(
        cramers_v_vs_target(df, cramers_cols).to_frame(),
        tables_dir / "eda_cramers_v.csv",
    )

    # ------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------
    report.figures["target_distribution"] = _save_fig(
        figures_dir / "eda_target_distribution.png",
        plot_target_distribution,
        df=df,
        figsize=(6.5, 2.8),
    )
    report.figures["missing_values"] = _save_fig(
        figures_dir / "eda_missing_values.png",
        plot_missing_bar,
        df=df,
        figsize=_FIG_SIZE_SMALL,
    )
    report.figures["tenure_histogram"] = _save_fig(
        figures_dir / "eda_tenure_histogram.png",
        plot_numeric_histogram,
        df=df,
        column="tenure",
        figsize=_FIG_SIZE_SMALL,
    )
    report.figures["monthly_charges_histogram"] = _save_fig(
        figures_dir / "eda_monthly_charges_histogram.png",
        plot_numeric_histogram,
        df=df,
        column="MonthlyCharges",
        figsize=_FIG_SIZE_SMALL,
    )
    report.figures["total_charges_box"] = _save_fig(
        figures_dir / "eda_total_charges_box.png",
        plot_numeric_boxplot_by_target,
        df=df,
        column="TotalCharges",
        figsize=_FIG_SIZE_SMALL,
    )
    report.figures["churn_by_contract"] = _save_fig(
        figures_dir / "eda_churn_by_contract.png",
        plot_categorical_churn_rate,
        df=df,
        column="Contract",
        figsize=_FIG_SIZE_SMALL,
    )
    report.figures["churn_by_payment"] = _save_fig(
        figures_dir / "eda_churn_by_payment_method.png",
        plot_categorical_churn_rate,
        df=df,
        column="PaymentMethod",
        figsize=_FIG_SIZE_WIDE,
    )
    report.figures["churn_by_tenure_bucket"] = _save_fig(
        figures_dir / "eda_churn_by_tenure_bucket.png",
        plot_categorical_churn_rate,
        df=df,
        column="tenure_bucket",
        figsize=_FIG_SIZE_SMALL,
    )
    report.figures["correlation_heatmap"] = _save_fig(
        figures_dir / "eda_correlation_heatmap.png",
        _plot_corr_wrapper,
        df=df,
        figsize=(7.0, 5.5),
    )

    log.info(
        "eda_report_generated",
        figures=len(report.figures),
        tables=len(report.tables),
        figures_dir=str(figures_dir),
        tables_dir=str(tables_dir),
    )
    return report


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _plot_corr_wrapper(df: pd.DataFrame, *, ax: Axes) -> None:
    corr = numeric_correlation(
        df,
        columns=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "services_count",
            "avg_monthly_spend",
            "monthly_spend_gap",
        ],
    )
    plot_correlation_heatmap(corr, ax=ax)


def _save_fig(
    path: Path,
    plotter: Callable[..., Any],
    *,
    figsize: tuple[float, float],
    **kwargs: Any,
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    try:
        plotter(ax=ax, **kwargs)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _dump_csv(df_out: pd.DataFrame, path: Path, *, index: bool = True) -> Path:
    df_out.to_csv(path, index=index)
    return path
