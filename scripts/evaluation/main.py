"""Punto de entrada oficial de la Fase 3 (TDSP · evaluación).

Implementa el entregable **"Reporte del modelo final"** exigido por la
rúbrica del Módulo 6 del Diplomado MLDS. Genera, para el modelo ganador
registrado en ``models/``:

* Métricas en ``val`` (a threshold por defecto y a threshold sintonizado).
* Matriz de confusión.
* Curva de calibración.
* Tabla y figura del barrido de threshold.
* (Opcional) Importancia de features (RF / coef LR / `feature_importances_`
  de LightGBM).

Por convención del proyecto, **el conjunto de prueba (`test.parquet`) no
se evalúa aquí** — está reservado para la Fase 4. Usa `--include-test`
solo si necesitas un avance preliminar.

Uso:

```bash
python scripts/evaluation/main.py
make evaluate
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from churnlens.config import Settings
from churnlens.features.preprocessing import TARGET_COL
from churnlens.logger import get_logger
from churnlens.models.evaluation import (
    binary_metrics,
    optimal_threshold,
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sweep,
    save_metrics_json,
    threshold_sweep,
)
from churnlens.models.registry import best_model_by, list_models, load_model
from churnlens.models.train import _predict_proba

log = get_logger("scripts.evaluation")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Genera el reporte de evaluación del modelo ganador (Fase 3).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Nombre del modelo. Default: el de mayor PR-AUC en val.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Evalúa también sobre test.parquet (uso reservado a Fase 4).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity a WARNING.",
    )
    return parser.parse_args(argv)


def _feature_importance(model: object, feature_names: list[str]) -> pd.Series | None:
    """Extrae una serie de importancias del modelo, si está disponible."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(
            np.asarray(model.feature_importances_),
            index=feature_names,
            name="importance",
        )
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        return pd.Series(np.abs(coef), index=feature_names, name="|coef|")
    return None


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script de evaluación."""
    args = _parse_args(argv)
    settings = Settings(log_level="WARNING") if args.quiet else Settings()

    figures_dir = settings.project_root / "reports" / "figures"
    tables_dir = settings.project_root / "reports" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        model_name = args.model
    else:
        best = best_model_by("pr_auc", split="val")
        if best is None:
            log.error("evaluation_no_models", available=[e.name for e in list_models()])
            return 1
        model_name = best.name
    log.info("phase3_evaluation_start", model=model_name)

    model, metadata = load_model(model_name)
    feature_set = list(metadata.get("feature_set") or [])

    val_df = pd.read_parquet(settings.processed_dir / "val.parquet")
    y_val = val_df[TARGET_COL].astype("int8").to_numpy()
    x_val = val_df[feature_set].astype("float32").to_numpy()
    proba_val = _predict_proba(model, x_val)

    choice = optimal_threshold(y_val, proba_val, metric="f1")
    metrics_default = binary_metrics(y_val, proba_val, threshold=0.5)
    metrics_tuned = binary_metrics(y_val, proba_val, threshold=choice.threshold)
    sweep = threshold_sweep(y_val, proba_val)

    report: dict[str, object] = {
        "model": model_name,
        "feature_set_size": len(feature_set),
        "threshold_default": 0.5,
        "threshold_tuned": choice.threshold,
        "metrics_val_default": metrics_default,
        "metrics_val_tuned": metrics_tuned,
    }

    sweep_path = tables_dir / f"evaluation_threshold_sweep_{model_name}.csv"
    sweep.to_csv(sweep_path, index_label="threshold")

    plot_threshold_sweep(
        sweep,
        out_path=figures_dir / f"evaluation_threshold_{model_name}.png",
        chosen=choice.threshold,
        title=f"Threshold sweep — {model_name} (val)",
    )
    plot_pr_curves(
        {model_name: (y_val, proba_val)},
        out_path=figures_dir / f"evaluation_pr_{model_name}.png",
        title=f"PR curve — {model_name} (val)",
    )
    plot_roc_curves(
        {model_name: (y_val, proba_val)},
        out_path=figures_dir / f"evaluation_roc_{model_name}.png",
        title=f"ROC curve — {model_name} (val)",
    )
    plot_calibration(
        y_val,
        proba_val,
        out_path=figures_dir / f"evaluation_calibration_{model_name}.png",
        title=f"Calibración — {model_name} (val)",
    )
    plot_confusion_matrix(
        y_val,
        (proba_val >= choice.threshold).astype("int8"),
        out_path=figures_dir / f"evaluation_confusion_{model_name}.png",
        title=f"Matriz de confusión — {model_name} (val · thr={choice.threshold:.3f})",
    )

    importance = _feature_importance(model, feature_set)
    if importance is not None:
        plot_feature_importance(
            importance,
            out_path=figures_dir / f"evaluation_importance_{model_name}.png",
            title=f"Importancia de features — {model_name}",
        )
        importance.sort_values(ascending=False).to_csv(
            tables_dir / f"evaluation_importance_{model_name}.csv",
            index_label="feature",
            header=["importance"],
        )

    if args.include_test:
        log.warning("phase3_test_evaluated", note="Fase 4 — uso anticipado del held-out.")
        test_df = pd.read_parquet(settings.processed_dir / "test.parquet")
        y_test = test_df[TARGET_COL].astype("int8").to_numpy()
        x_test = test_df[feature_set].astype("float32").to_numpy()
        proba_test = _predict_proba(model, x_test)
        metrics_test = binary_metrics(y_test, proba_test, threshold=choice.threshold)
        report["metrics_test_tuned"] = metrics_test

    summary_path = tables_dir / f"evaluation_summary_{model_name}.json"
    save_metrics_json(report, summary_path)
    log.info("phase3_evaluation_done", summary=str(summary_path))

    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
