"""Visualizaciones reutilizables para el EDA.

Las funciones reciben siempre un :class:`matplotlib.axes.Axes` opcional —
si no se provee, crean uno nuevo. Esto permite componer figuras complejas
desde notebooks o desde el orquestador `report.generate_eda_report`.

Todas las funciones devuelven el ``Axes`` para encadenamiento.

Convenciones de estilo:

* Paleta neutra ``"Set2"`` para categóricas, ``"viridis"`` para correlaciones.
* Fuente sans-serif (default matplotlib).
* No se llama a ``plt.show()`` — el renderizado queda a discreción del caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from churnlens.eda.summary import churn_rate_by_category
from churnlens.features.preprocessing import TARGET_COL, binarize_target

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Estilo base aplicado al importar el módulo. Es idempotente.
sns.set_theme(style="whitegrid", context="notebook", palette="Set2")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"


def plot_target_distribution(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL,
    ax: Axes | None = None,
) -> Axes:
    """Barra horizontal con la distribución del target y % anotado."""
    ax = ax or plt.subplots(figsize=(6, 2.5))[1]
    series = df[target].astype("string")
    counts = series.value_counts(dropna=False).sort_index()
    total = counts.sum()
    colors = sns.color_palette("Set2", n_colors=len(counts))
    bars = ax.barh(counts.index.tolist(), counts.values, color=colors)
    for bar, count in zip(bars, counts.values, strict=False):
        pct = count / total
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1%})",
            va="center",
            fontsize=10,
        )
    ax.set_xlim(0, total * 1.18)
    ax.set_xlabel("Clientes")
    ax.set_title(f"Distribución de {target}")
    return ax


def plot_numeric_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    hue: str | None = TARGET_COL,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Histograma de una variable numérica, opcionalmente segmentado por target."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    data = df[[column, hue]].dropna() if hue else df[[column]].dropna()
    sns.histplot(
        data=data,
        x=column,
        hue=hue,
        bins=bins,
        kde=True,
        common_norm=False,
        stat="density",
        ax=ax,
    )
    ax.set_title(f"Distribución de {column}" + (f" segmentada por {hue}" if hue else ""))
    return ax


def plot_numeric_boxplot_by_target(
    df: pd.DataFrame,
    column: str,
    *,
    target: str = TARGET_COL,
    ax: Axes | None = None,
) -> Axes:
    """Boxplot de una numérica contra el target."""
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    sns.boxplot(data=df, x=target, y=column, ax=ax)
    ax.set_title(f"{column} por {target}")
    return ax


def plot_categorical_churn_rate(
    df: pd.DataFrame,
    column: str,
    *,
    target: str = TARGET_COL,
    ax: Axes | None = None,
    overall_rate: float | None = None,
) -> Axes:
    """Barra horizontal con tasa de churn por categoría y línea base global."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    summary = churn_rate_by_category(df, column, target=target)
    rates = summary["churn_rate"]
    base = overall_rate if overall_rate is not None else float(binarize_target(df[target]).mean())
    bars = ax.barh(rates.index.astype(str), rates.values, color=sns.color_palette("Set2"))
    for bar, val, count in zip(bars, rates.values, summary["count"], strict=False):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}  (n={count:,})",
            va="center",
            fontsize=9,
        )
    ax.axvline(base, color="crimson", linestyle="--", linewidth=1.2, label=f"Global = {base:.1%}")
    ax.set_xlim(0, max(rates.max() + 0.18, base + 0.18))
    ax.set_xlabel("Tasa de churn")
    ax.set_title(f"Churn rate por {column}")
    ax.legend(loc="lower right", frameon=False)
    return ax


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str = "Correlación (Spearman)",
) -> Axes:
    """Heatmap triangular sobre una matriz de correlación pre-calculada."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title(title)
    return ax


def plot_missing_bar(df: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Barra horizontal con el conteo de valores faltantes por columna."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    missing = df.isna().sum().sort_values(ascending=True)
    missing = missing[missing > 0]
    if missing.empty:
        ax.text(
            0.5,
            0.5,
            "Sin valores faltantes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        return ax
    ax.barh(missing.index.tolist(), missing.values, color=sns.color_palette("Set2"))
    ax.set_xlabel("Filas con NaN")
    ax.set_title("Valores faltantes por columna")
    for i, v in enumerate(missing.values):
        ax.text(v + 0.1, i, str(int(v)), va="center", fontsize=9)
    return ax
