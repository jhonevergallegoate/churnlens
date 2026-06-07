"""Auditoría de equidad algorítmica sobre el conjunto held-out (Fase 5).

Implementa el análisis comprometido en
``docs/governance/ethics_and_fairness.md`` (§3): para cada atributo
sensible se calculan métricas por subgrupo y los indicadores agregados
de disparidad, comparados contra los umbrales declarados en ese
documento.

Métricas por subgrupo:

* ``selection_rate`` — fracción del grupo predicha como _churner_ por
  encima del threshold operativo.
* ``tpr`` (recall) / ``fpr`` — componentes de _equalized odds_.
* ``precision`` — calidad de los contactos disparados sobre el grupo.
* ``ece`` — _expected calibration error_ del subgrupo (bins uniformes).

Indicadores agregados por atributo:

* **Disparate Impact (DI)** — razón ``min/max`` de tasas de selección
  entre grupos. La regla del 80 % (EEOC) exige DI ≥ 0.80; calcular la
  razón en dirección ``min/max`` hace el chequeo simétrico, equivalente
  al rango [0.80, 1.25] documentado.
* **Demographic Parity difference (DPD)** — ``max − min`` de tasas de
  selección.
* **Equalized Odds difference (EOD)** — máximo entre la brecha de TPR y
  la brecha de FPR.
* **Max ECE** — peor calibración entre subgrupos.

Decisiones:

* Los atributos sensibles **no salen del parquet transformado**: se
  reconstruyen desde el dataset crudo replicando el split determinista
  de la Fase 2 (semilla 42). La alineación se verifica comparando la
  secuencia completa del target contra ``test.parquet`` — cualquier
  divergencia aborta la auditoría.
* ``SeniorCitizen`` **no es feature del modelo** (el
  ``ColumnTransformer`` lo descarta vía ``remainder="drop"``); se audita
  igualmente porque el modelo puede discriminarlo a través de _proxies_.
* La auditoría es **informativa, no bloqueante**: reporta veredictos por
  umbral y deja la decisión de mitigación al proceso documentado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from churnlens.config import Settings
from churnlens.config import settings as default_settings
from churnlens.logger import get_logger

log = get_logger(__name__)

SENSITIVE_ATTRIBUTES: Final[tuple[str, ...]] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
)
"""Atributos auditados — los comprometidos en `ethics_and_fairness.md` §2."""

DISPARATE_IMPACT_MIN: Final[float] = 0.80
"""Regla del 80 % de la EEOC (equivalente simétrico del rango [0.80, 1.25])."""

EQUALIZED_ODDS_MAX: Final[float] = 0.10
DEMOGRAPHIC_PARITY_MAX: Final[float] = 0.10
ECE_MAX: Final[float] = 0.05


@dataclass(frozen=True)
class FairnessAuditResult:
    """Resultado completo de :func:`run_fairness_audit`."""

    model_name: str
    threshold: float
    n_rows: int
    groups: pd.DataFrame
    summary: dict[str, dict[str, Any]]
    groups_path: Path
    summary_path: Path
    figure_path: Path

    @property
    def all_within_thresholds(self) -> bool:
        """``True`` si ningún atributo viola los umbrales declarados."""
        return all(entry["within_thresholds"] for entry in self.summary.values())


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def expected_calibration_error(
    y_true: NDArray[Any] | pd.Series,
    y_proba: NDArray[Any] | pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """Calcula el ECE con bins uniformes sobre [0, 1].

    ECE = Σ_b (n_b / N) · |frecuencia observada_b − probabilidad media_b|.

    Args:
        y_true: etiquetas binarias.
        y_proba: probabilidades de la clase positiva.
        n_bins: número de bins uniformes.

    Returns:
        ECE en [0, 1]; 0 indica calibración perfecta.
    """
    y_t = np.asarray(y_true).astype("float64")
    y_p = np.asarray(y_proba).astype("float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # El bin superior es inclusivo para capturar p == 1.0.
    bin_idx = np.clip(np.digitize(y_p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = len(y_p)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_t[mask].mean() - y_p[mask].mean())
    return float(ece)


def group_metrics(
    y_true: NDArray[Any] | pd.Series,
    y_proba: NDArray[Any] | pd.Series,
    groups: pd.Series,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Calcula las métricas de fairness por valor del atributo sensible.

    Args:
        y_true: etiquetas binarias del split auditado.
        y_proba: probabilidades de la clase positiva.
        groups: serie con el valor del atributo sensible por fila.
        threshold: punto de corte operativo.

    Returns:
        DataFrame indexado por valor del grupo con columnas ``n``,
        ``prevalence``, ``selection_rate``, ``tpr``, ``fpr``,
        ``precision`` y ``ece``. ``tpr``/``fpr``/``precision`` son
        ``NaN`` cuando el denominador del grupo es cero.
    """
    y_t = np.asarray(y_true).astype("int8")
    y_p = np.asarray(y_proba).astype("float64")
    if not (len(y_t) == len(y_p) == len(groups)):
        msg = (
            "y_true, y_proba y groups deben tener la misma longitud: "
            f"{len(y_t)}, {len(y_p)}, {len(groups)}."
        )
        raise ValueError(msg)
    y_hat = (y_p >= threshold).astype("int8")
    labels = groups.astype(str).to_numpy()

    rows = []
    for value in sorted(pd.unique(labels)):
        mask = labels == value
        t, p, hat = y_t[mask], y_p[mask], y_hat[mask]
        positives = int(t.sum())
        negatives = int((1 - t).sum())
        predicted_pos = int(hat.sum())
        rows.append(
            {
                "group": value,
                "n": int(mask.sum()),
                "prevalence": float(t.mean()),
                "selection_rate": float(hat.mean()),
                "tpr": float(hat[t == 1].mean()) if positives else float("nan"),
                "fpr": float(hat[t == 0].mean()) if negatives else float("nan"),
                "precision": float(t[hat == 1].mean()) if predicted_pos else float("nan"),
                "ece": expected_calibration_error(t, p),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def fairness_summary(group_df: pd.DataFrame) -> dict[str, Any]:
    """Agrega las métricas por grupo en los indicadores de disparidad.

    Args:
        group_df: salida de :func:`group_metrics` para **un** atributo.

    Returns:
        Dict con ``disparate_impact``, ``demographic_parity_diff``,
        ``equalized_odds_diff``, ``max_ece``, los veredictos individuales
        y el veredicto agregado ``within_thresholds``.
    """
    sel = group_df["selection_rate"]
    di = float(sel.min() / sel.max()) if sel.max() > 0 else float("nan")
    dpd = float(sel.max() - sel.min())
    tpr_gap = float(group_df["tpr"].max() - group_df["tpr"].min())
    fpr_gap = float(group_df["fpr"].max() - group_df["fpr"].min())
    gaps = [gap for gap in (tpr_gap, fpr_gap) if not np.isnan(gap)]
    eod = max(gaps) if gaps else float("nan")
    max_ece = float(group_df["ece"].max())

    di_ok = bool(di >= DISPARATE_IMPACT_MIN) if not np.isnan(di) else False
    dpd_ok = bool(dpd < DEMOGRAPHIC_PARITY_MAX)
    eod_ok = bool(eod < EQUALIZED_ODDS_MAX) if not np.isnan(eod) else False
    ece_ok = bool(max_ece < ECE_MAX)

    return {
        "groups": {str(g): int(n) for g, n in group_df["n"].items()},
        "disparate_impact": di,
        "disparate_impact_ok": di_ok,
        "demographic_parity_diff": dpd,
        "demographic_parity_ok": dpd_ok,
        "equalized_odds_diff": eod,
        "equalized_odds_ok": eod_ok,
        "max_ece": max_ece,
        "max_ece_ok": ece_ok,
        "within_thresholds": di_ok and dpd_ok and eod_ok and ece_ok,
    }


def audit_attributes(
    sensitive: pd.DataFrame,
    y_true: NDArray[Any] | pd.Series,
    y_proba: NDArray[Any] | pd.Series,
    *,
    threshold: float,
    attributes: tuple[str, ...] = SENSITIVE_ATTRIBUTES,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Audita varios atributos sensibles y consolida tabla + resumen.

    Args:
        sensitive: DataFrame con una columna por atributo sensible,
            alineado fila a fila con ``y_true``/``y_proba``.
        y_true: etiquetas binarias.
        y_proba: probabilidades de la clase positiva.
        threshold: punto de corte operativo.
        attributes: columnas de ``sensitive`` a auditar.

    Returns:
        Tupla ``(tabla, resumen)``: tabla larga indexada por
        ``(attribute, group)`` y dict ``{atributo: fairness_summary}``.
    """
    missing = [a for a in attributes if a not in sensitive.columns]
    if missing:
        msg = f"Atributos sensibles ausentes del DataFrame: {missing}."
        raise ValueError(msg)

    tables: list[pd.DataFrame] = []
    summary: dict[str, dict[str, Any]] = {}
    for attr in attributes:
        per_group = group_metrics(y_true, y_proba, sensitive[attr], threshold=threshold)
        summary[attr] = fairness_summary(per_group)
        per_group = per_group.assign(attribute=attr).set_index("attribute", append=True)
        tables.append(per_group.reorder_levels(["attribute", "group"]))
    return pd.concat(tables), summary


# ---------------------------------------------------------------------------
# Reconstrucción de atributos sensibles del held-out
# ---------------------------------------------------------------------------
def reconstruct_sensitive_test(
    *,
    settings: Settings | None = None,
    attributes: tuple[str, ...] = SENSITIVE_ATTRIBUTES,
) -> tuple[pd.DataFrame, NDArray[Any]]:
    """Reconstruye los atributos sensibles crudos del split de test.

    Replica exactamente la secuencia de la Fase 2
    (:func:`churnlens.features.pipeline.run_preprocessing`): carga
    validada → features derivadas → drop ``customerID`` → split
    estratificado con ``settings.random_seed``. Como el split es
    determinista, las filas resultantes son las mismas que produjeron
    ``test.parquet``.

    Args:
        settings: configuración del proyecto.
        attributes: columnas sensibles a extraer.

    Returns:
        Tupla ``(sensitive, y_test)`` con índice reseteado, alineada con
        el orden de filas de ``test.parquet``.
    """
    from churnlens.data.loader import TelcoChurnLoader
    from churnlens.features.engineering import add_engineered_features
    from churnlens.features.splits import stratified_split

    settings = settings or default_settings
    df = TelcoChurnLoader(settings=settings).load_validated()
    df = add_engineered_features(df).drop(columns=["customerID"])
    split = stratified_split(df, random_state=settings.random_seed)
    sensitive = split.X_test[list(attributes)].reset_index(drop=True)
    return sensitive, split.y_test.to_numpy().astype("int8")


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
def plot_fairness_groups(
    table: pd.DataFrame,
    *,
    out_path: Path | str,
    threshold: float,
    title: str = "Auditoría de fairness por subgrupo",
) -> Path:
    """Dibuja selection rate / TPR / FPR / precision por subgrupo y atributo.

    Args:
        table: tabla larga de :func:`audit_attributes` (índice
            ``(attribute, group)``).
        out_path: ruta PNG destino.
        threshold: threshold operativo (para el subtítulo).
        title: título de la figura.

    Returns:
        Ruta del PNG escrito.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    attributes = list(table.index.get_level_values("attribute").unique())
    metrics = ("selection_rate", "tpr", "fpr", "precision")
    colors = ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")

    fig, axes = plt.subplots(
        1, len(attributes), figsize=(3.4 * len(attributes), 4.4), dpi=120, sharey=True
    )
    axes_list = np.atleast_1d(axes)
    for ax, attr in zip(axes_list, attributes, strict=False):
        sub = table.loc[attr]
        x = np.arange(len(sub.index))
        width = 0.8 / len(metrics)
        for i, (metric, color) in enumerate(zip(metrics, colors, strict=False)):
            ax.bar(x + i * width, sub[metric].to_numpy(), width, label=metric, color=color)
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(sub.index.astype(str), fontsize=8)
        ax.set_title(attr, fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes_list[0].set_ylabel("score")
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(metrics), fontsize=8)
    fig.suptitle(f"{title} (threshold = {threshold:.2f})", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------
def run_fairness_audit(
    *,
    model_name: str = "logreg_l1",
    threshold: float | None = None,
    settings: Settings | None = None,
    attributes: tuple[str, ...] = SENSITIVE_ATTRIBUTES,
) -> FairnessAuditResult:
    """Ejecuta la auditoría de fairness completa sobre ``test.parquet``.

    Args:
        model_name: modelo registrado a auditar.
        threshold: punto de corte; default = el sintonizado del manifest.
        settings: configuración del proyecto.
        attributes: atributos sensibles a auditar.

    Returns:
        :class:`FairnessAuditResult` con tabla, resumen y rutas persistidas.

    Raises:
        FileNotFoundError: si faltan el modelo o ``test.parquet``.
        RuntimeError: si el split reconstruido no se alinea con el parquet.
    """
    from churnlens.features.preprocessing import TARGET_COL
    from churnlens.models.registry import load_model

    settings = settings or default_settings
    model, metadata = load_model(model_name, models_dir=settings.models_dir)
    feature_set = list(metadata.get("feature_set") or [])
    if threshold is None:
        threshold = float(
            ((metadata.get("metrics") or {}).get("val_tuned") or {}).get("threshold", 0.5)
        )

    test_path = settings.processed_dir / "test.parquet"
    if not test_path.exists():
        msg = f"No existe {test_path}. Ejecuta el preprocesamiento (Fase 2)."
        raise FileNotFoundError(msg)
    test_df = pd.read_parquet(test_path)
    y_test = test_df[TARGET_COL].astype("int8").to_numpy()
    x_test = test_df[feature_set].astype("float32").to_numpy()

    sensitive, y_reconstructed = reconstruct_sensitive_test(
        settings=settings, attributes=attributes
    )
    if len(sensitive) != len(test_df) or not np.array_equal(y_reconstructed, y_test):
        msg = (
            "El split reconstruido no coincide con test.parquet "
            f"({len(sensitive)} vs {len(test_df)} filas). Verifica que los "
            "datos crudos y la semilla no hayan cambiado desde la Fase 2."
        )
        raise RuntimeError(msg)

    proba = _predict_proba(model, x_test)
    table, summary = audit_attributes(
        sensitive, y_test, proba, threshold=threshold, attributes=attributes
    )

    figures_dir = settings.project_root / "reports" / "figures"
    tables_dir = settings.project_root / "reports" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    groups_path = tables_dir / f"fairness_groups_{model_name}.csv"
    table.to_csv(groups_path)
    summary_path = tables_dir / f"fairness_summary_{model_name}.json"
    payload: dict[str, Any] = {
        "model": model_name,
        "split": "test",
        "threshold": threshold,
        "n_rows": len(test_df),
        "thresholds_reference": {
            "disparate_impact_min": DISPARATE_IMPACT_MIN,
            "demographic_parity_max": DEMOGRAPHIC_PARITY_MAX,
            "equalized_odds_max": EQUALIZED_ODDS_MAX,
            "ece_max": ECE_MAX,
        },
        "attributes": summary,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    figure_path = plot_fairness_groups(
        table,
        out_path=figures_dir / f"fairness_audit_{model_name}.png",
        threshold=threshold,
        title=f"Fairness — {model_name} (test)",
    )

    log.info(
        "fairness_audit_done",
        model=model_name,
        threshold=threshold,
        within_thresholds={a: s["within_thresholds"] for a, s in summary.items()},
        summary=str(summary_path),
    )
    return FairnessAuditResult(
        model_name=model_name,
        threshold=threshold,
        n_rows=len(test_df),
        groups=table,
        summary=summary,
        groups_path=groups_path,
        summary_path=summary_path,
        figure_path=figure_path,
    )


def _predict_proba(estimator: Any, x: NDArray[Any]) -> NDArray[Any]:
    """Probabilidad de la clase positiva (espejo de `models.train`).

    Se replica aquí para que el módulo de fairness no importe
    ``churnlens.models.train`` (que arrastra LightGBM a nivel de módulo).
    """
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x))[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x))
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.full(scores.shape[0], 0.5)
        return np.asarray((scores - s_min) / (s_max - s_min))
    return np.asarray(estimator.predict(x), dtype="float64")
