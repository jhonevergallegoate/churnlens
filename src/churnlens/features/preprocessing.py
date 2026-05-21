"""Pipeline de preprocesamiento reproducible para modelado.

Este módulo expone la función `build_preprocessor`, que arma un
`sklearn.compose.ColumnTransformer` con los siguientes bloques:

1. **Numéricas continuas** → imputación por mediana + `StandardScaler`.
2. **Ordinal `Contract`** → `OrdinalEncoder` con orden explícito.
3. **Binarias `Yes/No` + `gender`** → `OrdinalEncoder` con orden explícito.
4. **Nominales multi-clase** → `OneHotEncoder(drop='first')`.

Decisiones documentadas en `docs/data/data_quality_report.md` y
`docs/data/data_summary_report.md`.

El transformador es:

* **Idempotente**: dos `fit` consecutivos sobre los mismos datos producen
  artefactos equivalentes.
* **Serializable**: puede persistirse con `joblib` para inferencia.
* **Sin leakage**: cualquier estadístico (`median`, `mean`, `std`,
  categorías) se aprende solo del conjunto de entrenamiento.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from churnlens.data.schema import (
    CONTRACT_VALUES,
    GENDER_VALUES,
    INTERNET_ADDON_VALUES,
    INTERNET_SERVICE_VALUES,
    MULTIPLE_LINES_VALUES,
    PAYMENT_METHOD_VALUES,
    YES_NO,
)
from churnlens.features.engineering import TENURE_BUCKET_LABELS

# ---------------------------------------------------------------------------
# Columnas por bloque
# ---------------------------------------------------------------------------
TARGET_COL: Final[str] = "Churn"

NUMERIC_COLS: Final[tuple[str, ...]] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)
"""Numéricas continuas / discretas del dataset original."""

ENGINEERED_NUMERIC_COLS: Final[tuple[str, ...]] = (
    "services_count",
    "avg_monthly_spend",
    "monthly_spend_gap",
)
"""Numéricas producidas por `add_engineered_features`."""

ORDINAL_COLS: Final[tuple[str, ...]] = (
    "Contract",
    "tenure_bucket",
)
"""Categóricas con orden inherente."""

BINARY_CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
)
"""Categóricas binarias — se codifican con `OrdinalEncoder` (orden fijo)."""

_BOOLEAN_ENGINEERED_COLS: Final[tuple[str, ...]] = (
    "has_internet",
    "has_phone",
    "auto_payment",
)
"""Booleanos producidos por `add_engineered_features`."""

NOMINAL_CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaymentMethod",
)
"""Categóricas multi-clase sin orden inherente."""

# ---------------------------------------------------------------------------
# Catálogos de categorías (deben coincidir con `data.schema`)
# ---------------------------------------------------------------------------
_BINARY_CATEGORIES: Final[dict[str, list[str]]] = {
    "gender": GENDER_VALUES,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "PaperlessBilling": YES_NO,
}

_ORDINAL_CATEGORIES: Final[dict[str, list[str]]] = {
    "Contract": CONTRACT_VALUES,
    "tenure_bucket": list(TENURE_BUCKET_LABELS),
}

_NOMINAL_CATEGORIES: Final[dict[str, list[str]]] = {
    "MultipleLines": MULTIPLE_LINES_VALUES,
    "InternetService": INTERNET_SERVICE_VALUES,
    "OnlineSecurity": INTERNET_ADDON_VALUES,
    "OnlineBackup": INTERNET_ADDON_VALUES,
    "DeviceProtection": INTERNET_ADDON_VALUES,
    "TechSupport": INTERNET_ADDON_VALUES,
    "StreamingTV": INTERNET_ADDON_VALUES,
    "StreamingMovies": INTERNET_ADDON_VALUES,
    "PaymentMethod": PAYMENT_METHOD_VALUES,
}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def binarize_target(y: pd.Series) -> pd.Series:
    """Convierte la columna ``Churn`` (`Yes`/`No`) en `int8` (1/0).

    Args:
        y: serie original con valores `"Yes"` o `"No"`.

    Returns:
        Serie de enteros con el mismo índice; `Yes` → 1, `No` → 0.

    Raises:
        ValueError: si aparecen valores fuera del dominio esperado.
    """
    values = y.astype(str)
    invalid = set(values.unique()).difference({"Yes", "No"})
    if invalid:
        msg = f"Valores fuera de dominio en Churn: {sorted(invalid)}"
        raise ValueError(msg)
    return values.map({"Yes": 1, "No": 0}).astype("int8")


def build_preprocessor(*, include_engineered: bool = True) -> ColumnTransformer:
    """Arma el `ColumnTransformer` canónico del proyecto.

    Args:
        include_engineered: si ``True`` (default), incluye las columnas
            derivadas (`tenure_bucket`, `services_count`, etc.) en el
            transformador. Para activarlas, las features deben haber sido
            añadidas previamente con
            :func:`churnlens.features.engineering.add_engineered_features`.

    Returns:
        `ColumnTransformer` no ajustado, listo para ``fit`` sobre el
        conjunto de entrenamiento.
    """
    numeric_cols: list[str] = list(NUMERIC_COLS)
    ordinal_cols: list[str] = list(_filtered_ordinal_cols(include_engineered))
    ordinal_categories: list[list[object]] = [list(_ORDINAL_CATEGORIES[c]) for c in ordinal_cols]
    binary_cols: list[str] = list(BINARY_CATEGORICAL_COLS)
    binary_categories: list[list[object]] = [list(_BINARY_CATEGORIES[c]) for c in binary_cols]
    nominal_cols: list[str] = list(NOMINAL_CATEGORICAL_COLS)
    nominal_categories: list[list[object]] = [list(_NOMINAL_CATEGORIES[c]) for c in nominal_cols]

    if include_engineered:
        numeric_cols.extend(ENGINEERED_NUMERIC_COLS)
        binary_cols.extend(_BOOLEAN_ENGINEERED_COLS)
        binary_categories.extend([[False, True] for _ in _BOOLEAN_ENGINEERED_COLS])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    categories=binary_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    nominal_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    categories=nominal_categories,
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("bin", binary_pipeline, binary_cols),
            ("nom", nominal_pipeline, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _filtered_ordinal_cols(include_engineered: bool) -> tuple[str, ...]:
    """Devuelve las columnas ordinales activas según se incluyan derivadas."""
    if include_engineered:
        return ORDINAL_COLS
    return tuple(c for c in ORDINAL_COLS if c != "tenure_bucket")
