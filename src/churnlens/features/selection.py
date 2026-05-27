"""Selección y extracción de características para el dataset preprocesado.

Este módulo opera sobre la matriz ya transformada por
:func:`churnlens.features.preprocessing.build_preprocessor` y proporciona
**cuatro técnicas complementarias** de scoring de features, más una rutina
de **consolidación por consenso**:

1. **Mutual Information** (`mutual_info_classif`) — filtro univariado no
   lineal. Captura dependencias de cualquier forma funcional entre cada
   feature y la variable objetivo.
2. **χ²** (`chi2`) — filtro univariado aplicado únicamente al subconjunto
   de features no negativas (ordinales, binarias, one-hot). El bloque
   numérico se omite porque tras ``StandardScaler`` contiene valores
   negativos que invalidan el test.
3. **L1 logistic regression** — selección _embedded_: el coeficiente
   absoluto de un modelo lineal con penalización L1 y
   ``class_weight='balanced'`` mide la utilidad lineal de cada feature
   bajo el sesgo del clasificador.
4. **Permutation importance sobre Random Forest** — selección _wrapper_
   no lineal y agnóstica al modelo final. Mide la caída de PR-AUC al
   permutar cada feature.

Las cuatro técnicas se ejecutan sobre el conjunto de entrenamiento;
cada una produce un ranking. ``consensus_top_k`` cuenta cuántas técnicas
incluyen a cada feature en su top-k y rompe empates por _score_ medio
normalizado.

Justificación del diseño:

* **Filtros + embedded + wrapper** cubren los tres grandes paradigmas de
  selección y se complementan: un filtro detecta señal univariada, un
  modelo lineal captura efectos marginales, y el _wrapper_ no lineal
  revela interacciones que los anteriores pueden ignorar.
* **Sin _data leakage_**: todo se calcula sobre `train.parquet`. El
  ``val`` y ``test`` permanecen ciegos en esta etapa.
* **Reproducibilidad bit-exacta**: cada estimador recibe la semilla
  global del proyecto (``settings.random_seed``, default 42).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from churnlens.config import settings as default_settings
from churnlens.features.preprocessing import TARGET_COL
from churnlens.logger import get_logger

log = get_logger(__name__)

DEFAULT_TOP_K: int = 20
DEFAULT_RF_ESTIMATORS: int = 200
DEFAULT_PERMUTATION_REPEATS: int = 10
DEFAULT_PERMUTATION_SCORING: str = "average_precision"
"""Métrica usada para la permutation importance — PR-AUC (apropiada para
clase desbalanceada del 26.5 %)."""


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Resultado consolidado de las cuatro técnicas de selección.

    Attributes:
        scores: tabla _wide_ con una columna por técnica
            (`mutual_info`, `chi2`, `l1_logreg`, `permutation_rf`).
            Las features sin score válido (p. ej. χ² no aplica a las
            features numéricas escaladas) aparecen como ``NaN``.
        ranks: tabla _wide_ con el ranking (1 = mejor) por técnica.
        consensus: DataFrame con `feature`, `votes`, `mean_score`,
            ordenada por (votes desc, mean_score desc).
        top_k: features seleccionadas por consenso (la lista final).
        k: tamaño del top-k usado para el voto de cada técnica.
        meta: información de reproducibilidad (semilla, hiperparámetros).
    """

    scores: pd.DataFrame
    ranks: pd.DataFrame
    consensus: pd.DataFrame
    top_k: list[str]
    k: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Empaqueta el resultado como dict JSON-serializable."""
        return {
            "k": self.k,
            "top_k_features": self.top_k,
            "meta": self.meta,
            "consensus": self.consensus.to_dict(orient="records"),
            "scores": self.scores.reset_index().to_dict(orient="records"),
            "ranks": self.ranks.reset_index().to_dict(orient="records"),
        }


# ---------------------------------------------------------------------------
# Carga del conjunto de entrenamiento preprocesado
# ---------------------------------------------------------------------------
def load_training_matrix(
    train_path: Path | str,
    *,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carga ``train.parquet`` y separa features y target.

    Args:
        train_path: ruta al parquet de entrenamiento producido por la
            Fase 2.
        target_col: nombre de la columna objetivo.

    Returns:
        Tupla ``(X, y)`` donde ``X`` es un DataFrame ``float32`` con las
        35 features y ``y`` es una serie ``int8`` con la etiqueta binaria.
    """
    df = pd.read_parquet(train_path)
    if target_col not in df.columns:
        msg = f"Falta la columna objetivo '{target_col}' en {train_path}."
        raise ValueError(msg)
    y = df[target_col].astype("int8")
    x = df.drop(columns=[target_col]).astype("float32")
    return x, y


# ---------------------------------------------------------------------------
# Técnicas individuales
# ---------------------------------------------------------------------------
def mutual_information_scores(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
) -> pd.Series:
    """Devuelve la _mutual information_ entre cada feature y `y`.

    Filtro no paramétrico válido para variables continuas y discretas.
    sklearn calcula la MI estimada vía k-NN (Kraskov et al.).

    Returns:
        Serie indexada por feature con la MI en _nats_. Valores ≥ 0.
    """
    scores = mutual_info_classif(
        x.to_numpy(dtype="float64"),
        y.to_numpy(dtype="int8"),
        random_state=random_state,
    )
    return pd.Series(scores, index=x.columns, name="mutual_info").sort_values(ascending=False)


def chi2_scores(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Devuelve el estadístico χ² para features no negativas.

    El test χ² requiere features con valores ≥ 0. Tras ``StandardScaler``
    el bloque numérico (`tenure`, `MonthlyCharges`, `TotalCharges`,
    `services_count`, `avg_monthly_spend`, `monthly_spend_gap`) puede ser
    negativo, así que esas features se ignoran y aparecen como ``NaN``.

    Returns:
        Serie indexada por feature; ``NaN`` para features no aplicables.
    """
    is_non_negative = (x.min(axis=0) >= 0.0).to_numpy()
    cols_ok = x.columns[is_non_negative].tolist()
    if not cols_ok:
        return pd.Series(np.nan, index=x.columns, name="chi2")

    stats, _pvals = chi2(x[cols_ok].to_numpy(dtype="float64"), y.to_numpy(dtype="int8"))
    raw = pd.Series(stats, index=cols_ok, name="chi2")
    full = pd.Series(np.nan, index=x.columns, name="chi2")
    full.loc[raw.index] = raw.to_numpy()
    return full.sort_values(ascending=False, na_position="last")


def l1_logistic_importance(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    c: float = 0.5,
    random_state: int = 42,
) -> pd.Series:
    """Importancia _embedded_ vía Logistic Regression con penalización L1.

    Args:
        x: matriz de features ya preprocesadas (`float32`).
        y: etiquetas binarias `int8` alineadas con `x`.
        c: inverso de la fuerza de regularización (sklearn ``C``). Valores
            pequeños (más regularización) producen modelos más esparsos.
        random_state: semilla del solver.

    Returns:
        Serie de coeficientes absolutos ordenados de mayor a menor.
    """
    model = LogisticRegression(
        penalty="l1",
        C=c,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    model.fit(x.to_numpy(dtype="float64"), y.to_numpy(dtype="int8"))
    coefs = np.abs(model.coef_).ravel()
    return pd.Series(coefs, index=x.columns, name="l1_logreg").sort_values(ascending=False)


def permutation_importance_rf(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    n_estimators: int = DEFAULT_RF_ESTIMATORS,
    n_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = 42,
    scoring: str = DEFAULT_PERMUTATION_SCORING,
    n_jobs: int = -1,
) -> pd.Series:
    """Importancia por permutación sobre un Random Forest.

    Ajusta un RandomForest balanced y mide la caída media de
    ``scoring`` al permutar cada feature. A diferencia de la importancia
    Gini, no está sesgada hacia features con alta cardinalidad.

    Args:
        x: matriz de features ya preprocesadas (`float32`).
        y: etiquetas binarias `int8` alineadas con `x`.
        n_estimators: número de árboles del RF.
        n_repeats: número de permutaciones por feature.
        random_state: semilla del RF y del shuffle.
        scoring: métrica usada para medir la caída (``average_precision``
            por defecto = PR-AUC).
        n_jobs: ``-1`` para usar todos los cores disponibles.

    Returns:
        Serie con la _importance mean_ por feature (puede ser negativa
        si la permutación mejora el score, lo cual indica feature ruido).
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf.fit(x.to_numpy(dtype="float64"), y.to_numpy(dtype="int8"))
    result = permutation_importance(
        rf,
        x.to_numpy(dtype="float64"),
        y.to_numpy(dtype="int8"),
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    return pd.Series(result.importances_mean, index=x.columns, name="permutation_rf").sort_values(
        ascending=False
    )


# ---------------------------------------------------------------------------
# Consolidación por consenso
# ---------------------------------------------------------------------------
def _rank_descending(series: pd.Series) -> pd.Series:
    """Asigna rangos 1..N (1 = mayor) ignorando NaN (que reciben NaN)."""
    return series.rank(method="min", ascending=False, na_option="bottom")


def _normalize(series: pd.Series) -> pd.Series:
    """Reescala una serie al rango [0, 1] ignorando NaN."""
    s = series.astype("float64")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return s * 0.0
    return (s - mn) / (mx - mn)


def consensus_top_k(
    scores: dict[str, pd.Series],
    *,
    k: int = DEFAULT_TOP_K,
    feature_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Combina los rankings de varias técnicas en un consenso top-k.

    Args:
        scores: dict ``{nombre_tecnica: pd.Series}`` (índice = feature).
        k: tamaño del top de cada técnica para el voto.
        feature_order: orden canónico (default = unión de los índices).

    Returns:
        Tupla ``(scores_wide, ranks_wide, consensus_df, top_k_features)``:

        * ``scores_wide``: DataFrame con una columna por técnica.
        * ``ranks_wide``: DataFrame con el ranking por técnica.
        * ``consensus_df``: con columnas ``feature``, ``votes``,
          ``mean_norm_score``, ordenada para selección.
        * ``top_k_features``: lista de exactamente ``k`` features.
    """
    if not scores:
        msg = "El dict de scores no puede estar vacío."
        raise ValueError(msg)

    feature_order = feature_order or sorted({f for s in scores.values() for f in s.index})
    scores_wide = pd.DataFrame({name: s.reindex(feature_order) for name, s in scores.items()})
    ranks_wide = scores_wide.apply(_rank_descending)
    norm_wide = scores_wide.apply(_normalize)

    # Voto: 1 si la feature está en el top-k de la técnica, 0 si no.
    votes_wide = (ranks_wide <= k).astype("int8")
    consensus = pd.DataFrame(
        {
            "feature": feature_order,
            "votes": votes_wide.sum(axis=1).to_numpy(),
            "mean_norm_score": norm_wide.mean(axis=1, skipna=True).to_numpy(),
        }
    )
    consensus = consensus.sort_values(
        by=["votes", "mean_norm_score"], ascending=[False, False], ignore_index=True
    )
    top_k_features = consensus.head(k)["feature"].tolist()
    return scores_wide, ranks_wide, consensus, top_k_features


# ---------------------------------------------------------------------------
# API pública de alto nivel
# ---------------------------------------------------------------------------
def run_feature_selection(
    train_path: Path | str | None = None,
    *,
    k: int = DEFAULT_TOP_K,
    random_state: int | None = None,
    rf_estimators: int = DEFAULT_RF_ESTIMATORS,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    permutation_n_jobs: int = -1,
) -> FeatureSelectionResult:
    """Orquesta las cuatro técnicas sobre el conjunto de entrenamiento.

    Args:
        train_path: ruta a ``train.parquet``. Si es ``None`` se usa el
            default del proyecto (``data/processed/train.parquet``).
        k: tamaño del consenso top-k.
        random_state: semilla (default = ``settings.random_seed``).
        rf_estimators: árboles del RF en permutation importance.
        permutation_repeats: repeticiones del shuffle.
        permutation_n_jobs: ``-1`` para usar todos los cores.

    Returns:
        :class:`FeatureSelectionResult` con todos los rankings.
    """
    seed = random_state if random_state is not None else default_settings.random_seed
    path = Path(train_path) if train_path else default_settings.processed_dir / "train.parquet"
    if not path.exists():
        msg = (
            f"No existe {path}. Ejecuta primero `churnlens preprocess run` "
            "para producir el conjunto de entrenamiento."
        )
        raise FileNotFoundError(msg)

    x, y = load_training_matrix(path)
    log.info("feature_selection_loaded", n_rows=len(x), n_features=x.shape[1])

    mi = mutual_information_scores(x, y, random_state=seed)
    log.info("feature_selection_mi_done", top=mi.head(3).to_dict())
    ch = chi2_scores(x, y)
    log.info("feature_selection_chi2_done", top=ch.dropna().head(3).to_dict())
    l1 = l1_logistic_importance(x, y, random_state=seed)
    log.info("feature_selection_l1_done", top=l1.head(3).to_dict())
    perm = permutation_importance_rf(
        x,
        y,
        n_estimators=rf_estimators,
        n_repeats=permutation_repeats,
        random_state=seed,
        n_jobs=permutation_n_jobs,
    )
    log.info("feature_selection_perm_done", top=perm.head(3).to_dict())

    scores_wide, ranks_wide, consensus, top_k = consensus_top_k(
        {"mutual_info": mi, "chi2": ch, "l1_logreg": l1, "permutation_rf": perm},
        k=k,
        feature_order=list(x.columns),
    )
    log.info("feature_selection_consensus", k=k, top_k=top_k)

    return FeatureSelectionResult(
        scores=scores_wide,
        ranks=ranks_wide,
        consensus=consensus,
        top_k=top_k,
        k=k,
        meta={
            "random_state": seed,
            "rf_estimators": rf_estimators,
            "permutation_repeats": permutation_repeats,
            "permutation_scoring": DEFAULT_PERMUTATION_SCORING,
            "n_features_total": int(x.shape[1]),
            "techniques": ["mutual_info", "chi2", "l1_logreg", "permutation_rf"],
        },
    )


def persist_feature_selection(
    result: FeatureSelectionResult,
    *,
    tables_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Persiste el resultado a tablas CSV + manifiesto JSON.

    Genera:
    * ``reports/tables/feature_selection_scores.csv``
    * ``reports/tables/feature_selection_ranks.csv``
    * ``reports/tables/feature_selection_consensus.csv``
    * ``data/processed/feature_consensus.json``

    Args:
        result: salida de :func:`run_feature_selection`.
        tables_dir: directorio destino para las tablas CSV.
        processed_dir: directorio destino para el manifiesto JSON.

    Returns:
        Dict con las rutas escritas.
    """
    tables_dir_path = (
        Path(tables_dir) if tables_dir else default_settings.project_root / "reports" / "tables"
    )
    processed_dir_path = Path(processed_dir) if processed_dir else default_settings.processed_dir
    tables_dir_path.mkdir(parents=True, exist_ok=True)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    scores_path = tables_dir_path / "feature_selection_scores.csv"
    ranks_path = tables_dir_path / "feature_selection_ranks.csv"
    consensus_path = tables_dir_path / "feature_selection_consensus.csv"
    manifest_path = processed_dir_path / "feature_consensus.json"

    result.scores.to_csv(scores_path, index_label="feature")
    result.ranks.to_csv(ranks_path, index_label="feature")
    result.consensus.to_csv(consensus_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "k": result.k,
                "top_k_features": result.top_k,
                "meta": result.meta,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "scores": scores_path,
        "ranks": ranks_path,
        "consensus": consensus_path,
        "manifest": manifest_path,
    }
