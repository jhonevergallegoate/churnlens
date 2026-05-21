"""Ingeniería y preprocesamiento de variables.

Este subpaquete contiene tres bloques:

* `engineering`: derivación de _features_ a partir del esquema crudo
  (`tenure_bucket`, `services_count`, `auto_payment`, etc.).
* `preprocessing`: ensamblado del `ColumnTransformer` de scikit-learn
  con imputación, codificación y escalamiento; reproducible y serializable.
* `splits`: partición estratificada train / val / test con semilla fija.
"""

from __future__ import annotations

from churnlens.features.engineering import (
    ADDON_COLS,
    SERVICE_COLS,
    TENURE_BUCKET_EDGES,
    TENURE_BUCKET_LABELS,
    add_engineered_features,
)
from churnlens.features.pipeline import PreprocessingArtifacts, run_preprocessing
from churnlens.features.preprocessing import (
    BINARY_CATEGORICAL_COLS,
    ENGINEERED_NUMERIC_COLS,
    NOMINAL_CATEGORICAL_COLS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    TARGET_COL,
    binarize_target,
    build_preprocessor,
)
from churnlens.features.splits import SplitResult, stratified_split

__all__ = [
    "ADDON_COLS",
    "BINARY_CATEGORICAL_COLS",
    "ENGINEERED_NUMERIC_COLS",
    "NOMINAL_CATEGORICAL_COLS",
    "NUMERIC_COLS",
    "ORDINAL_COLS",
    "SERVICE_COLS",
    "TARGET_COL",
    "TENURE_BUCKET_EDGES",
    "TENURE_BUCKET_LABELS",
    "PreprocessingArtifacts",
    "SplitResult",
    "add_engineered_features",
    "binarize_target",
    "build_preprocessor",
    "run_preprocessing",
    "stratified_split",
]
