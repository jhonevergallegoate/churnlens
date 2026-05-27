"""Evaluación de modelos binarios y _threshold tuning_.

Las funciones de este módulo asumen el contrato típico de scikit-learn:

* ``y_true``: array binario ``{0, 1}``.
* ``y_proba``: array de probabilidades de la clase positiva en ``[0, 1]``.

Decisiones:

* La **métrica primaria** del proyecto es **PR-AUC** (``average_precision``).
  Apropiada para clase desbalanceada (~26.5 % churn) porque no se ve
  inflada por la fácil clasificación de los negativos.
* El **threshold** por defecto se ajusta maximizando F1 sobre el conjunto
  de validación. F1 balancea precision y recall y refleja el objetivo
  típico de retención: maximizar las cancelaciones evitadas sin
  desperdiciar contactos sobre cuentas que no iban a churnar.
* Las figuras se renderizan con backend ``Agg`` (sin pantalla) y se
  guardan como PNG idempotente.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(t, 3) for t in np.linspace(0.05, 0.95, 91))
"""Resolución de 0.01 en [0.05, 0.95] para el barrido de threshold."""

_EPS = 1e-12


@dataclass(frozen=True)
class ThresholdChoice:
    """Resultado de :func:`optimal_threshold`."""

    metric: str
    threshold: float
    score: float


# ---------------------------------------------------------------------------
# Métricas escalares
# ---------------------------------------------------------------------------
def binary_metrics(
    y_true: NDArray[Any] | pd.Series,
    y_proba: NDArray[Any] | pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Devuelve el bundle estándar de métricas para clasificación binaria.

    Métricas:

    * ``pr_auc`` — PR-AUC (average precision). **Métrica primaria.**
    * ``roc_auc`` — Área bajo ROC.
    * ``brier`` — error cuadrático medio de la probabilidad (calibración).
    * ``f1`` — F1 a `threshold`.
    * ``precision`` — Precision a `threshold`.
    * ``recall`` — Recall a `threshold`.
    * ``accuracy`` — Exactitud a `threshold`.
    * ``positive_rate`` — Fracción de predicciones positivas a `threshold`.
    * ``base_rate`` — Tasa real de positivos en ``y_true``.
    * ``lift_at_10`` — _Lift_ del decil top: precision en el 10 % con
      mayor probabilidad dividido por la base rate.

    Args:
        y_true: vector de etiquetas binarias.
        y_proba: vector de probabilidades de la clase positiva.
        threshold: punto de corte para las métricas que lo requieren.

    Returns:
        Dict ``{nombre_métrica: float}``.
    """
    y_t = np.asarray(y_true).astype("int8")
    y_p = np.asarray(y_proba).astype("float64")
    y_hat = (y_p >= threshold).astype("int8")

    base_rate = float(y_t.mean())
    return {
        "pr_auc": float(average_precision_score(y_t, y_p)),
        "roc_auc": float(roc_auc_score(y_t, y_p)),
        "brier": float(brier_score_loss(y_t, y_p)),
        "f1": float(f1_score(y_t, y_hat, zero_division=0)),
        "precision": float(precision_score(y_t, y_hat, zero_division=0)),
        "recall": float(recall_score(y_t, y_hat, zero_division=0)),
        "accuracy": float((y_t == y_hat).mean()),
        "positive_rate": float(y_hat.mean()),
        "base_rate": base_rate,
        "lift_at_10": _lift_at_top_decile(y_t, y_p, fraction=0.10, base_rate=base_rate),
        "threshold": float(threshold),
    }


def _lift_at_top_decile(
    y_true: NDArray[Any],
    y_proba: NDArray[Any],
    *,
    fraction: float,
    base_rate: float,
) -> float:
    """Calcula el _lift_ del top `fraction` (precision_top / base_rate)."""
    if base_rate <= 0:
        return 0.0
    n = max(round(len(y_proba) * fraction), 1)
    top_idx = np.argsort(y_proba)[-n:]
    return float(y_true[top_idx].mean() / base_rate)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
def threshold_sweep(
    y_true: NDArray[Any] | pd.Series,
    y_proba: NDArray[Any] | pd.Series,
    *,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Evalúa métricas en una grilla de thresholds.

    Returns:
        DataFrame indexado por threshold con columnas:
        ``precision``, ``recall``, ``f1``, ``accuracy``, ``positive_rate``.
    """
    y_t = np.asarray(y_true).astype("int8")
    y_p = np.asarray(y_proba).astype("float64")
    rows = []
    for thr in thresholds:
        y_hat = (y_p >= thr).astype("int8")
        rows.append(
            {
                "threshold": round(float(thr), 4),
                "precision": float(precision_score(y_t, y_hat, zero_division=0)),
                "recall": float(recall_score(y_t, y_hat, zero_division=0)),
                "f1": float(f1_score(y_t, y_hat, zero_division=0)),
                "accuracy": float((y_t == y_hat).mean()),
                "positive_rate": float(y_hat.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


def optimal_threshold(
    y_true: NDArray[Any] | pd.Series,
    y_proba: NDArray[Any] | pd.Series,
    *,
    metric: str = "f1",
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
) -> ThresholdChoice:
    """Devuelve el threshold que maximiza `metric` sobre la grilla.

    Args:
        y_true: etiquetas binarias.
        y_proba: probabilidades de la clase positiva.
        metric: ``"f1"`` (default), ``"precision"``, ``"recall"``,
            ``"accuracy"``.
        thresholds: grilla a evaluar.

    Returns:
        :class:`ThresholdChoice` con el threshold ganador y su score.
    """
    sweep = threshold_sweep(y_true, y_proba, thresholds=thresholds)
    if metric not in sweep.columns:
        msg = f"métrica '{metric}' no soportada — usa una de {list(sweep.columns)}"
        raise ValueError(msg)
    series = sweep[metric]
    best = series.idxmax()
    return ThresholdChoice(metric=metric, threshold=float(best), score=float(series.loc[best]))


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_pr_curves(
    runs: dict[str, tuple[NDArray[Any], NDArray[Any]]],
    *,
    out_path: Path | str,
    title: str = "Curvas Precision-Recall",
) -> Path:
    """Dibuja PR curves para un conjunto de modelos.

    Args:
        runs: dict ``{nombre_modelo: (y_true, y_proba)}``.
        out_path: ruta PNG destino.
        title: título de la figura.

    Returns:
        Ruta del PNG escrito.
    """
    out = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    for name, (y_t, y_p) in runs.items():
        precision, recall, _thr = precision_recall_curve(y_t, y_p)
        ap = average_precision_score(y_t, y_p)
        ax.plot(recall, precision, label=f"{name} · AP={ap:.3f}")
    base_rate = float(np.mean(next(iter(runs.values()))[0]))
    ax.axhline(
        base_rate, color="grey", linestyle="--", linewidth=1, label=f"base rate = {base_rate:.3f}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_roc_curves(
    runs: dict[str, tuple[NDArray[Any], NDArray[Any]]],
    *,
    out_path: Path | str,
    title: str = "Curvas ROC",
) -> Path:
    """Dibuja ROC curves para un conjunto de modelos."""
    out = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    for name, (y_t, y_p) in runs.items():
        fpr, tpr, _thr = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, label=f"{name} · AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="azar")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusion_matrix(
    y_true: NDArray[Any] | pd.Series,
    y_pred: NDArray[Any] | pd.Series,
    *,
    out_path: Path | str,
    title: str = "Matriz de confusión",
    labels: tuple[str, str] = ("No churn", "Churn"),
) -> Path:
    """Genera una matriz de confusión etiquetada."""
    out = _ensure_dir(Path(out_path))
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j, i, str(v), ha="center", va="center", color="black" if v < cm.max() / 2 else "white"
        )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_calibration(
    y_true: NDArray[Any] | pd.Series,
    y_proba: NDArray[Any] | pd.Series,
    *,
    out_path: Path | str,
    n_bins: int = 10,
    title: str = "Curva de calibración",
) -> Path:
    """Dibuja la curva de calibración (reliability diagram)."""
    out = _ensure_dir(Path(out_path))
    prob_true, prob_pred = calibration_curve(
        np.asarray(y_true), np.asarray(y_proba), n_bins=n_bins, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfecto")
    ax.plot(prob_pred, prob_true, marker="o", label="modelo")
    ax.set_xlabel("Probabilidad predicha (media del bin)")
    ax.set_ylabel("Frecuencia observada de positivos")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_feature_importance(
    importance: pd.Series,
    *,
    out_path: Path | str,
    top_n: int = 20,
    title: str = "Top features por importancia",
) -> Path:
    """Dibuja un bar plot horizontal con las top-N features."""
    out = _ensure_dir(Path(out_path))
    top = importance.dropna().sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(top))), dpi=120)
    ax.barh(top.index.astype(str), top.to_numpy(), color="#1f77b4")
    ax.set_xlabel("importancia")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_threshold_sweep(
    sweep: pd.DataFrame,
    *,
    out_path: Path | str,
    chosen: float | None = None,
    title: str = "Threshold sweep — Precision / Recall / F1",
) -> Path:
    """Dibuja precision/recall/F1 vs threshold y marca el threshold elegido."""
    out = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    for col, color in (("precision", "#d62728"), ("recall", "#2ca02c"), ("f1", "#1f77b4")):
        ax.plot(sweep.index, sweep[col], label=col, color=color)
    if chosen is not None:
        ax.axvline(
            chosen,
            color="black",
            linestyle=":",
            linewidth=1,
            label=f"threshold elegido = {chosen:.3f}",
        )
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def save_metrics_json(metrics: dict[str, Any], path: Path | str) -> Path:
    """Persiste un dict de métricas como JSON serializable."""
    out = _ensure_dir(Path(path))
    out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return out
