"""Orquestador de entrenamiento, comparación y persistencia (Fase 3).

La función pública :func:`train_models` ejecuta el siguiente protocolo:

1. Carga ``train.parquet`` y ``val.parquet`` desde ``data/processed/``.
2. Si se proporciona ``feature_subset``, restringe ``X`` a esas columnas.
3. Para cada modelo de :data:`MODEL_SPECS` (o el subconjunto solicitado):

   a. **Validación cruzada** estratificada 5-fold sobre ``train`` con
      PR-AUC y ROC-AUC.
   b. **Fit final** sobre todo ``train``.
   c. **Threshold tuning** maximizando F1 sobre ``val``.
   d. **Métricas** en ``train`` y ``val`` (a threshold 0.5 y a threshold
      sintonizado).
   e. **Persistencia** con :func:`churnlens.models.registry.save_model`.

4. Construye una tabla comparativa, identifica el ganador por PR-AUC y
   persiste los artefactos (CV scores, summary, threshold sweep del
   ganador).

Las dependencias externas son ``scikit-learn`` y ``lightgbm``; ambas son
parte de la matriz de dependencias estándar (Python 3.10–3.12). El
random state global proviene de ``settings.random_seed``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churnlens.config import settings as default_settings
from churnlens.features.preprocessing import TARGET_COL
from churnlens.logger import get_logger
from churnlens.models.baseline import build_baseline_estimators
from churnlens.models.evaluation import (
    binary_metrics,
    optimal_threshold,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sweep,
    save_metrics_json,
    threshold_sweep,
)
from churnlens.models.registry import ModelEntry, save_model
from churnlens.utils.hashing import compute_sha256

log = get_logger(__name__)

DEFAULT_CV_FOLDS: int = 5
DEFAULT_SCORE_KEY: str = "pr_auc"

# ---------------------------------------------------------------------------
# Catálogo de modelos
# ---------------------------------------------------------------------------
ModelFactory = Callable[[int], Any]


def _logreg_balanced(seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )


def _logreg_l1(seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l1",
        C=0.5,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )


def _random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _hist_gb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=seed,
    )


def _lightgbm(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _dummy_stratified(seed: int) -> DummyClassifier:
    return DummyClassifier(strategy="stratified", random_state=seed)


def _dummy_most_frequent(_: int) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def _dummy_prior(_: int) -> DummyClassifier:
    return DummyClassifier(strategy="prior")


MODEL_SPECS: dict[str, ModelFactory] = {
    "dummy_stratified": _dummy_stratified,
    "dummy_most_frequent": _dummy_most_frequent,
    "dummy_prior": _dummy_prior,
    "logreg_balanced": _logreg_balanced,
    "logreg_l1": _logreg_l1,
    "random_forest": _random_forest,
    "hist_gb": _hist_gb,
    "lightgbm": _lightgbm,
}
"""Catálogo de modelos disponibles para `train_models`."""


# ---------------------------------------------------------------------------
# Resultado de entrenamiento
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingArtifacts:
    """Artefactos producidos por :func:`train_models`."""

    summary_table: pd.DataFrame
    cv_table: pd.DataFrame
    threshold_sweeps: dict[str, pd.DataFrame]
    model_entries: dict[str, ModelEntry]
    best_model_name: str
    paths: dict[str, Path] = field(default_factory=dict)

    @property
    def best_entry(self) -> ModelEntry:
        """Atajo al :class:`ModelEntry` del modelo ganador."""
        return self.model_entries[self.best_model_name]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _load_xy(parquet_path: Path, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(parquet_path)
    if target_col not in df.columns:
        msg = f"Falta '{target_col}' en {parquet_path}."
        raise ValueError(msg)
    y = df[target_col].astype("int8")
    x = df.drop(columns=[target_col]).astype("float32")
    return x, y


def _serialize_hparams(estimator: Any) -> dict[str, Any]:
    """Convierte get_params() en un dict JSON-serializable."""
    params = estimator.get_params(deep=False)
    out: dict[str, Any] = {}
    for k, v in params.items():
        try:
            if v is None or isinstance(v, (bool, int, float, str)):
                out[k] = v
            else:
                out[k] = str(v)
        except Exception:  # pragma: no cover - defensa contra params exóticos
            out[k] = repr(v)
    return out


def _predict_proba(estimator: Any, x: np.ndarray) -> np.ndarray:
    """Devuelve la probabilidad de la clase positiva si está disponible.

    Para modelos sin ``predict_proba`` cae a ``decision_function`` o a
    ``predict``. Útil para que los ``DummyClassifier`` también participen.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        return np.asarray(proba)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x))
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.full(scores.shape[0], 0.5)
        return np.asarray((scores - s_min) / (s_max - s_min))
    return np.asarray(estimator.predict(x), dtype="float64")


def _cv_pr_auc(
    estimator: Any, x: np.ndarray, y: np.ndarray, *, cv: int, seed: int
) -> tuple[float, float, float, float]:
    """Devuelve (pr_auc_mean, pr_auc_std, roc_auc_mean, roc_auc_std) en CV.

    La CV corre con ``n_jobs=1`` (folds en serie) porque los modelos del
    catálogo (RF, HGB, LightGBM) ya paralelizan internamente con
    ``n_jobs=-1``. Si la CV también pidiera todos los cores, se crearían
    ``n_cores × n_cores`` workers y el sistema entra en _live-lock_ por
    over-subscription (LightGBM es particularmente sensible a esto).
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pr = cross_val_score(estimator, x, y, scoring="average_precision", cv=splitter, n_jobs=1)
        roc = cross_val_score(estimator, x, y, scoring="roc_auc", cv=splitter, n_jobs=1)
    return float(pr.mean()), float(pr.std()), float(roc.mean()), float(roc.std())


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def train_models(  # noqa: PLR0915 - orquestador legible end-to-end, no se divide
    *,
    train_path: Path | str | None = None,
    val_path: Path | str | None = None,
    models: list[str] | None = None,
    feature_subset: list[str] | None = None,
    cv: int = DEFAULT_CV_FOLDS,
    random_state: int | None = None,
    models_dir: Path | str | None = None,
    tables_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
) -> TrainingArtifacts:
    """Entrena y compara los modelos solicitados, guardando artefactos.

    Args:
        train_path: ruta a ``train.parquet``. Default = ``data/processed``.
        val_path: ruta a ``val.parquet``. Default = ``data/processed``.
        models: subconjunto de :data:`MODEL_SPECS` a entrenar. Si es
            ``None``, se entrenan todos.
        feature_subset: si se entrega, restringe ``X`` a esas columnas
            (útil para entrenar con el top-k de
            :mod:`churnlens.features.selection`). También se entrenan los
            modelos con el set completo para tener una referencia.
        cv: número de folds en la CV estratificada.
        random_state: semilla (default = ``settings.random_seed``).
        models_dir: directorio destino de los ``.joblib``.
        tables_dir: directorio destino de las tablas CSV.
        figures_dir: directorio destino de las figuras PNG.

    Returns:
        :class:`TrainingArtifacts` con todos los resultados.
    """
    seed = random_state if random_state is not None else default_settings.random_seed
    train_p = Path(train_path) if train_path else default_settings.processed_dir / "train.parquet"
    val_p = Path(val_path) if val_path else default_settings.processed_dir / "val.parquet"
    if not train_p.exists() or not val_p.exists():
        msg = (
            f"No se encontraron los parquet de entrenamiento ({train_p}) "
            f"o validación ({val_p}). Ejecuta `churnlens preprocess run`."
        )
        raise FileNotFoundError(msg)

    models_dir_p = Path(models_dir) if models_dir else default_settings.project_root / "models"
    tables_dir_p = (
        Path(tables_dir) if tables_dir else default_settings.project_root / "reports" / "tables"
    )
    figures_dir_p = (
        Path(figures_dir) if figures_dir else default_settings.project_root / "reports" / "figures"
    )
    for d in (models_dir_p, tables_dir_p, figures_dir_p):
        d.mkdir(parents=True, exist_ok=True)

    x_train_full, y_train = _load_xy(train_p)
    x_val_full, y_val = _load_xy(val_p)

    if feature_subset is not None:
        missing = [c for c in feature_subset if c not in x_train_full.columns]
        if missing:
            msg = f"Las features {missing} no están en {train_p}."
            raise ValueError(msg)
        x_train = x_train_full[feature_subset]
        x_val = x_val_full[feature_subset]
    else:
        x_train = x_train_full
        x_val = x_val_full

    requested = models or list(MODEL_SPECS.keys())
    unknown = [m for m in requested if m not in MODEL_SPECS]
    if unknown:
        msg = f"Modelos desconocidos: {unknown}. Disponibles: {list(MODEL_SPECS)}"
        raise ValueError(msg)

    train_hash = compute_sha256(train_p)
    val_hash = compute_sha256(val_p)
    feature_set = list(x_train.columns)

    summary_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    sweeps: dict[str, pd.DataFrame] = {}
    entries: dict[str, ModelEntry] = {}
    pr_runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    x_train_arr = x_train.to_numpy(dtype="float32")
    y_train_arr = y_train.to_numpy(dtype="int8")
    x_val_arr = x_val.to_numpy(dtype="float32")
    y_val_arr = y_val.to_numpy(dtype="int8")

    for name in requested:
        factory = MODEL_SPECS[name]
        log.info("model_train_start", model=name, n_features=len(feature_set))
        estimator = factory(seed)

        try:
            cv_pr_mean, cv_pr_std, cv_roc_mean, cv_roc_std = _cv_pr_auc(
                factory(seed), x_train_arr, y_train_arr, cv=cv, seed=seed
            )
        except Exception as exc:  # pragma: no cover - defensa CV
            log.warning("cv_failed", model=name, error=str(exc))
            cv_pr_mean = cv_pr_std = cv_roc_mean = cv_roc_std = float("nan")

        cv_rows.append(
            {
                "model": name,
                "cv_pr_auc_mean": cv_pr_mean,
                "cv_pr_auc_std": cv_pr_std,
                "cv_roc_auc_mean": cv_roc_mean,
                "cv_roc_auc_std": cv_roc_std,
                "cv_folds": cv,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            estimator.fit(x_train_arr, y_train_arr)

        proba_train = _predict_proba(estimator, x_train_arr)
        proba_val = _predict_proba(estimator, x_val_arr)
        pr_runs[name] = (y_val_arr, proba_val)

        metrics_train_default = binary_metrics(y_train_arr, proba_train, threshold=0.5)
        metrics_val_default = binary_metrics(y_val_arr, proba_val, threshold=0.5)
        choice = optimal_threshold(y_val_arr, proba_val, metric="f1")
        metrics_val_tuned = binary_metrics(y_val_arr, proba_val, threshold=choice.threshold)
        sweep = threshold_sweep(y_val_arr, proba_val)
        sweeps[name] = sweep

        summary_rows.append(
            {
                "model": name,
                "algorithm": type(estimator).__name__,
                "n_features": len(feature_set),
                "cv_pr_auc_mean": cv_pr_mean,
                "cv_pr_auc_std": cv_pr_std,
                "cv_roc_auc_mean": cv_roc_mean,
                "train_pr_auc": metrics_train_default["pr_auc"],
                "val_pr_auc": metrics_val_default["pr_auc"],
                "val_roc_auc": metrics_val_default["roc_auc"],
                "val_brier": metrics_val_default["brier"],
                "val_f1@0.5": metrics_val_default["f1"],
                "val_threshold_tuned": choice.threshold,
                "val_f1_tuned": metrics_val_tuned["f1"],
                "val_precision_tuned": metrics_val_tuned["precision"],
                "val_recall_tuned": metrics_val_tuned["recall"],
                "val_lift_at_10": metrics_val_default["lift_at_10"],
            }
        )

        entry = save_model(
            estimator,
            name,
            metadata={
                "algorithm": type(estimator).__name__,
                "train_path": str(train_p),
                "val_path": str(val_p),
                "hash_train": train_hash,
                "hash_val": val_hash,
                "feature_set": feature_set,
                "hyperparameters": _serialize_hparams(estimator),
                "metrics": {
                    "train": metrics_train_default,
                    "val": metrics_val_default,
                    "val_tuned": metrics_val_tuned,
                    "cv": {
                        "pr_auc_mean": cv_pr_mean,
                        "pr_auc_std": cv_pr_std,
                        "roc_auc_mean": cv_roc_mean,
                        "roc_auc_std": cv_roc_std,
                        "n_folds": cv,
                    },
                },
                "random_state": seed,
                "feature_subset_applied": feature_subset is not None,
            },
            models_dir=models_dir_p,
        )
        entries[name] = entry
        log.info(
            "model_train_done",
            model=name,
            val_pr_auc=metrics_val_default["pr_auc"],
            cv_pr_auc=cv_pr_mean,
            threshold=choice.threshold,
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        by="val_pr_auc", ascending=False, ignore_index=True
    )
    cv_table = pd.DataFrame(cv_rows).sort_values(
        by="cv_pr_auc_mean", ascending=False, ignore_index=True
    )

    summary_path = tables_dir_p / "modeling_summary.csv"
    cv_path = tables_dir_p / "modeling_cv_scores.csv"
    summary.to_csv(summary_path, index=False)
    cv_table.to_csv(cv_path, index=False)

    pr_fig_path = plot_pr_curves(
        pr_runs,
        out_path=figures_dir_p / "modeling_pr_curves_val.png",
        title="PR curves — validación",
    )
    roc_fig_path = plot_roc_curves(
        pr_runs,
        out_path=figures_dir_p / "modeling_roc_curves_val.png",
        title="ROC curves — validación",
    )

    best_name = str(summary.iloc[0]["model"])
    best_sweep_path = tables_dir_p / f"modeling_threshold_sweep_{best_name}.csv"
    sweeps[best_name].to_csv(best_sweep_path, index_label="threshold")
    chosen_thr = float(summary.iloc[0]["val_threshold_tuned"])
    best_sweep_fig = plot_threshold_sweep(
        sweeps[best_name],
        out_path=figures_dir_p / f"modeling_threshold_sweep_{best_name}.png",
        chosen=chosen_thr,
        title=f"Threshold sweep — {best_name}",
    )

    summary_metrics = {
        "best_model": best_name,
        "score_key": DEFAULT_SCORE_KEY,
        "val_pr_auc_best": float(summary.iloc[0]["val_pr_auc"]),
        "val_threshold_tuned": chosen_thr,
        "n_features": len(feature_set),
        "feature_subset_applied": feature_subset is not None,
    }
    summary_json_path = tables_dir_p / "modeling_best_summary.json"
    save_metrics_json(summary_metrics, summary_json_path)

    return TrainingArtifacts(
        summary_table=summary,
        cv_table=cv_table,
        threshold_sweeps=sweeps,
        model_entries=entries,
        best_model_name=best_name,
        paths={
            "summary_csv": summary_path,
            "cv_csv": cv_path,
            "best_threshold_csv": best_sweep_path,
            "best_threshold_png": best_sweep_fig,
            "pr_curves_png": pr_fig_path,
            "roc_curves_png": roc_fig_path,
            "summary_json": summary_json_path,
            "models_dir": models_dir_p,
        },
    )


def quick_baseline_only(
    *,
    train_path: Path | str | None = None,
    val_path: Path | str | None = None,
    random_state: int | None = None,
    models_dir: Path | str | None = None,
    tables_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
) -> TrainingArtifacts:
    """Atajo: entrena únicamente los baselines del proyecto.

    Útil para tests y para reproducir el reporte de línea base sin
    pagar el costo de los modelos pesados.
    """
    return train_models(
        train_path=train_path,
        val_path=val_path,
        models=list(build_baseline_estimators().keys()),
        cv=DEFAULT_CV_FOLDS,
        random_state=random_state,
        models_dir=models_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )
