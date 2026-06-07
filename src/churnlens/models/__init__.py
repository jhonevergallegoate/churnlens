"""Subpaquete de modelado (Fase 3 del Diplomado MLDS).

Expone las APIs principales:

* :mod:`churnlens.models.baseline`     — referencias mínimas (DummyClassifier).
* :mod:`churnlens.models.registry`     — persistencia con metadatos auditables.
* :mod:`churnlens.models.evaluation`   — métricas, threshold tuning y figuras.
* :mod:`churnlens.models.train`        — orquestador end-to-end del entrenamiento.
* :mod:`churnlens.models.fairness`     — auditoría de equidad por subgrupos (Fase 5).

Convención: todos los entrenamientos consumen los `*.parquet` producidos por
la Fase 2 (`data/processed/`), nunca el CSV crudo. La estratificación, el
escalado y el codificado ya están aplicados; aquí solo se entrena, evalúa
y compara.

Nota de despliegue (Fase 4): los símbolos de :mod:`churnlens.models.train`
se cargan de forma **perezosa** (PEP 562) porque ese módulo importa LightGBM
y otras dependencias de entrenamiento que no existen (ni deben pesar) en la
imagen de producción — el serving solo necesita :mod:`registry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from churnlens.models.fairness import (
    SENSITIVE_ATTRIBUTES,
    FairnessAuditResult,
    run_fairness_audit,
)
from churnlens.models.registry import (
    ModelEntry,
    list_models,
    load_model,
    save_model,
)

if TYPE_CHECKING:
    from churnlens.models.train import (
        MODEL_SPECS,
        TrainingArtifacts,
        train_models,
    )

_TRAIN_EXPORTS = frozenset({"MODEL_SPECS", "TrainingArtifacts", "train_models"})


def __getattr__(name: str) -> Any:
    """Carga perezosa de `churnlens.models.train` (PEP 562)."""
    if name in _TRAIN_EXPORTS:
        from churnlens.models import train

        return getattr(train, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "BASELINE_MODEL_NAMES",
    "DEFAULT_THRESHOLDS",
    "MODEL_SPECS",
    "SENSITIVE_ATTRIBUTES",
    "FairnessAuditResult",
    "ModelEntry",
    "TrainingArtifacts",
    "binary_metrics",
    "build_baseline_estimators",
    "list_models",
    "load_model",
    "optimal_threshold",
    "run_fairness_audit",
    "save_model",
    "threshold_sweep",
    "train_models",
]
