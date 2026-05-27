"""Subpaquete de modelado (Fase 3 del Diplomado MLDS).

Expone las APIs principales:

* :mod:`churnlens.models.baseline`     — referencias mínimas (DummyClassifier).
* :mod:`churnlens.models.registry`     — persistencia con metadatos auditables.
* :mod:`churnlens.models.evaluation`   — métricas, threshold tuning y figuras.
* :mod:`churnlens.models.train`        — orquestador end-to-end del entrenamiento.

Convención: todos los entrenamientos consumen los `*.parquet` producidos por
la Fase 2 (`data/processed/`), nunca el CSV crudo. La estratificación, el
escalado y el codificado ya están aplicados; aquí solo se entrena, evalúa
y compara.
"""

from __future__ import annotations

from churnlens.models.baseline import (
    BASELINE_MODEL_NAMES,
    build_baseline_estimators,
)
from churnlens.models.evaluation import (
    DEFAULT_THRESHOLDS,
    binary_metrics,
    optimal_threshold,
    threshold_sweep,
)
from churnlens.models.registry import (
    ModelEntry,
    list_models,
    load_model,
    save_model,
)
from churnlens.models.train import (
    MODEL_SPECS,
    TrainingArtifacts,
    train_models,
)

__all__ = [
    "BASELINE_MODEL_NAMES",
    "DEFAULT_THRESHOLDS",
    "MODEL_SPECS",
    "ModelEntry",
    "TrainingArtifacts",
    "binary_metrics",
    "build_baseline_estimators",
    "list_models",
    "load_model",
    "optimal_threshold",
    "save_model",
    "threshold_sweep",
    "train_models",
]
