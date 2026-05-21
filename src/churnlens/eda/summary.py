"""Estadísticas descriptivas tabulares para el dataset Telco Customer Churn.

Cada función devuelve un :class:`pandas.DataFrame` o :class:`pandas.Series`
con tipos estables, lo que facilita exportar a Markdown, CSV o JSON sin
formateo adicional.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from churnlens.features.preprocessing import TARGET_COL, binarize_target

_PERCENTILES: Final[tuple[float, ...]] = (0.05, 0.25, 0.5, 0.75, 0.95)


def numeric_summary(df: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Resumen estadístico de columnas numéricas.

    Para cada columna devuelve: ``count``, ``missing``, ``mean``, ``std``,
    ``min``, percentiles 5/25/50/75/95, ``max``, ``skew`` y ``kurtosis``.

    Args:
        df: DataFrame de entrada.
        columns: subconjunto opcional de columnas. Si es ``None`` se usan
            todas las numéricas detectadas por pandas.

    Returns:
        DataFrame con una fila por variable.
    """
    cols = columns if columns is not None else df.select_dtypes("number").columns.tolist()
    rows: list[dict[str, object]] = []
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        row: dict[str, object] = {
            "variable": col,
            "count": int(non_null.size),
            "missing": int(series.isna().sum()),
            "mean": float(non_null.mean()) if non_null.size else float("nan"),
            "std": float(non_null.std(ddof=1)) if non_null.size > 1 else float("nan"),
            "min": float(non_null.min()) if non_null.size else float("nan"),
            "max": float(non_null.max()) if non_null.size else float("nan"),
            "skew": float(non_null.skew()) if non_null.size > 2 else float("nan"),
            "kurtosis": float(non_null.kurtosis()) if non_null.size > 3 else float("nan"),
        }
        for q in _PERCENTILES:
            row[f"p{int(q * 100):02d}"] = (
                float(non_null.quantile(q)) if non_null.size else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


def categorical_summary(df: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Resumen descriptivo de columnas categóricas / objeto.

    Para cada columna devuelve: ``n_unique``, ``missing``, ``top``
    (categoría más frecuente) y ``top_freq`` (su frecuencia relativa).
    """
    if columns is None:
        cols = df.select_dtypes(include=["category", "object", "string"]).columns.tolist()
    else:
        cols = list(columns)

    rows = []
    for col in cols:
        series = df[col].astype("string")
        counts = series.value_counts(dropna=True)
        top = str(counts.index[0]) if not counts.empty else ""
        top_freq = float(counts.iloc[0] / len(series)) if len(series) else float("nan")
        rows.append(
            {
                "variable": col,
                "n_unique": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
                "top": top,
                "top_freq": top_freq,
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def target_distribution(df: pd.DataFrame, *, target: str = TARGET_COL) -> pd.Series:
    """Distribución de la variable objetivo como conteos y porcentaje."""
    series = df[target].astype("string")
    counts = series.value_counts(dropna=False)
    pct = counts / counts.sum()
    out = pd.concat({"count": counts, "pct": pct}, axis=1)
    out.index.name = target
    return out


def churn_rate_by_category(
    df: pd.DataFrame,
    column: str,
    *,
    target: str = TARGET_COL,
    min_count: int = 30,
) -> pd.DataFrame:
    """Tasa de churn por nivel de una variable categórica.

    Args:
        df: DataFrame con la variable y el target.
        column: nombre de la columna categórica a analizar.
        target: nombre de la columna objetivo.
        min_count: tamaño mínimo de la categoría para considerarse confiable
            (categorías por debajo del umbral se conservan pero se marcan).

    Returns:
        DataFrame indexado por categoría con columnas ``count``, ``churn_rate``,
        ``is_reliable``.
    """
    y = (
        binarize_target(df[target])
        if df[target].dtype.name in {"category", "object", "string"}
        else df[target]
    )
    grouped = pd.DataFrame({"_cat": df[column].astype("string"), "_y": y.to_numpy()})
    summary = grouped.groupby("_cat", dropna=False).agg(
        count=("_y", "size"),
        churn_rate=("_y", "mean"),
    )
    summary["is_reliable"] = summary["count"] >= min_count
    summary.index.name = column
    return summary.sort_values("churn_rate", ascending=False)


def numeric_correlation(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """Matriz de correlación entre columnas numéricas.

    Args:
        df: DataFrame de entrada.
        columns: subconjunto opcional de columnas numéricas.
        method: método de correlación (``'pearson'``, ``'spearman'``,
            ``'kendall'``). Spearman por default por robustez a outliers.

    Returns:
        DataFrame cuadrado con la matriz de correlación.
    """
    cols = columns if columns is not None else df.select_dtypes("number").columns.tolist()
    return df[cols].corr(method=method)


def cramers_v_vs_target(
    df: pd.DataFrame,
    columns: list[str],
    *,
    target: str = TARGET_COL,
) -> pd.Series:
    """V de Cramér entre cada categórica y la variable objetivo.

    La V de Cramér es una medida de asociación entre dos variables
    categóricas, normalizada al rango ``[0, 1]``. Aplica corrección de
    Bergsma–Wicher para minimizar el sesgo en tablas pequeñas.
    """
    y = df[target].astype("string")
    results: dict[str, float] = {}
    for col in columns:
        contingency = pd.crosstab(df[col].astype("string"), y)
        results[col] = _cramers_v(contingency.to_numpy())
    return pd.Series(results, name="cramers_v").sort_values(ascending=False)


def _cramers_v(contingency: np.ndarray) -> float:
    """Implementación interna de la V de Cramér con corrección de sesgo."""
    chi2 = _chi_square(contingency)
    n = contingency.sum()
    if n == 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def _chi_square(contingency: np.ndarray) -> float:
    """Estadístico chi-cuadrado de Pearson sobre una tabla de contingencia."""
    row_totals = contingency.sum(axis=1, keepdims=True)
    col_totals = contingency.sum(axis=0, keepdims=True)
    total = contingency.sum()
    if total == 0:
        return 0.0
    expected = row_totals @ col_totals / total
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = (contingency - expected) ** 2 / expected
        diff = np.where(expected > 0, diff, 0.0)
    return float(diff.sum())
