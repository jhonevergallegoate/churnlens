"""Validaciones adicionales sobre el dataset cargado.

Las validaciones estructurales (tipos, dominios, dependencias) viven en
`schema.py` como esquema Pandera. Este módulo concentra **validaciones
de calidad** orientadas a alertar al usuario sobre anomalías que no son
estrictamente errores pero pueden afectar el modelado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churnlens.logger import get_logger

log = get_logger(__name__)


@dataclass
class QualityReport:
    """Reporte de calidad ligero del dataset."""

    n_rows: int
    n_cols: int
    n_duplicates: int
    target_positive_rate: float
    missing_per_column: dict[str, int] = field(default_factory=dict)
    constant_columns: list[str] = field(default_factory=list)

    @property
    def imbalance_ratio(self) -> float:
        """Razón negativos/positivos del target."""
        pos = self.target_positive_rate
        if pos in (0.0, 1.0):
            return float("inf")
        return (1.0 - pos) / pos


def compute_quality_report(df: pd.DataFrame, target_col: str = "Churn") -> QualityReport:
    """Calcula un reporte de calidad rápido sobre el DataFrame.

    Args:
        df:         DataFrame validado contra `RAW_SCHEMA`.
        target_col: Nombre de la columna objetivo.

    Returns:
        `QualityReport` con métricas clave.
    """
    missing_per_column = {
        col: int(n) for col, n in df.isna().sum().items() if int(n) > 0
    }

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    report = QualityReport(
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        n_duplicates=int(df.duplicated().sum()),
        target_positive_rate=float((df[target_col] == "Yes").mean()),
        missing_per_column=missing_per_column,
        constant_columns=constant_cols,
    )

    log.info(
        "quality_report",
        n_rows=report.n_rows,
        n_cols=report.n_cols,
        n_duplicates=report.n_duplicates,
        positive_rate=round(report.target_positive_rate, 4),
        n_constant_cols=len(constant_cols),
    )
    return report
