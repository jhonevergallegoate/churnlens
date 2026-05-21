"""Análisis exploratorio de datos reproducible.

Tres submódulos:

* `summary`: estadísticas descriptivas (numéricas, categóricas, target,
  correlaciones).
* `plots`: figuras matplotlib/seaborn reutilizables (univariado, bivariado,
  matriz de correlación).
* `report`: orquestador que genera todas las tablas y figuras del reporte
  oficial (`reports/figures/eda_*.png` y tablas exportadas).
"""

from __future__ import annotations

from churnlens.eda.plots import (
    plot_categorical_churn_rate,
    plot_correlation_heatmap,
    plot_missing_bar,
    plot_numeric_boxplot_by_target,
    plot_numeric_histogram,
    plot_target_distribution,
)
from churnlens.eda.report import EDAReport, generate_eda_report
from churnlens.eda.summary import (
    categorical_summary,
    churn_rate_by_category,
    numeric_correlation,
    numeric_summary,
    target_distribution,
)

__all__ = [
    "EDAReport",
    "categorical_summary",
    "churn_rate_by_category",
    "generate_eda_report",
    "numeric_correlation",
    "numeric_summary",
    "plot_categorical_churn_rate",
    "plot_correlation_heatmap",
    "plot_missing_bar",
    "plot_numeric_boxplot_by_target",
    "plot_numeric_histogram",
    "plot_target_distribution",
    "target_distribution",
]
