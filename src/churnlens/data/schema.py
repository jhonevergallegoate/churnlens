"""Esquema declarativo del dataset `Telco Customer Churn`.

El esquema se define con Pandera, lo que permite:

* Validar el dataframe **post-casteo** (`coerce=False`, dtypes exactos).
* Capturar errores de tipo, dominio, dependencias entre columnas y unicidad.
* Servir como **contrato** entre el productor (carga de datos) y los
  consumidores (EDA, modelado, despliegue).

Cualquier cambio en este archivo debe actualizar también
`docs/data/data_dictionary.md` y viceversa.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Dominios de las variables categóricas
# ---------------------------------------------------------------------------
YES_NO: Final[list[str]] = ["Yes", "No"]
GENDER_VALUES: Final[list[str]] = ["Female", "Male"]
MULTIPLE_LINES_VALUES: Final[list[str]] = ["No phone service", "No", "Yes"]
INTERNET_SERVICE_VALUES: Final[list[str]] = ["DSL", "Fiber optic", "No"]
INTERNET_ADDON_VALUES: Final[list[str]] = ["No internet service", "No", "Yes"]
CONTRACT_VALUES: Final[list[str]] = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_VALUES: Final[list[str]] = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# ---------------------------------------------------------------------------
# Checks reutilizables
# ---------------------------------------------------------------------------
def _check_phone_lines_consistency(df: pd.DataFrame) -> pd.Series:
    """Garantiza que `PhoneService = No` ⟹ `MultipleLines = 'No phone service'`."""
    mask_no_phone = df["PhoneService"].astype(str).eq("No")
    expected = df["MultipleLines"].astype(str).eq("No phone service")
    return (~mask_no_phone) | expected


def _check_internet_addons_consistency(df: pd.DataFrame) -> pd.Series:
    """Si `InternetService = No`, todos los add-ons deben ser 'No internet service'."""
    mask_no_internet = df["InternetService"].astype(str).eq("No")
    addon_cols = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    expected_addons = pd.concat(
        [df[c].astype(str).eq("No internet service") for c in addon_cols],
        axis=1,
    ).all(axis=1)
    return (~mask_no_internet) | expected_addons


def _cat_isin(values: list[str]) -> Check:
    """Construye un Check de pertenencia que funciona sobre Series categóricas."""
    allowed = tuple(values)

    def _check(s: pd.Series) -> pd.Series:
        return s.astype(str).isin(allowed)

    return Check(
        _check,
        name="in_allowed_domain",
        error=f"valor fuera del dominio permitido: {list(allowed)}",
    )


# ---------------------------------------------------------------------------
# Esquema principal
# ---------------------------------------------------------------------------
def build_raw_schema() -> DataFrameSchema:
    """Construye el esquema Pandera para el dataset post-casteo."""
    return DataFrameSchema(
        columns={
            "customerID": Column(
                "string",
                checks=Check.str_matches(r"^\d{4}-[A-Z]{5}$"),
                nullable=False,
                unique=True,
                description="Identificador único sintético del cliente.",
            ),
            "gender": Column(
                "category",
                checks=_cat_isin(GENDER_VALUES),
                nullable=False,
                description="Género auto-reportado del titular.",
            ),
            "SeniorCitizen": Column(
                "int8",
                checks=Check(lambda s: s.isin([0, 1]), name="senior_in_0_1"),
                nullable=False,
                description="1 si el cliente es senior (65+ años), 0 en caso contrario.",
            ),
            "Partner": Column(
                "category",
                checks=_cat_isin(YES_NO),
                nullable=False,
            ),
            "Dependents": Column(
                "category",
                checks=_cat_isin(YES_NO),
                nullable=False,
            ),
            "tenure": Column(
                "int16",
                checks=Check.in_range(0, 72, include_min=True, include_max=True),
                nullable=False,
                description="Antigüedad en meses desde el alta.",
            ),
            "PhoneService": Column(
                "category",
                checks=_cat_isin(YES_NO),
                nullable=False,
            ),
            "MultipleLines": Column(
                "category",
                checks=_cat_isin(MULTIPLE_LINES_VALUES),
                nullable=False,
            ),
            "InternetService": Column(
                "category",
                checks=_cat_isin(INTERNET_SERVICE_VALUES),
                nullable=False,
            ),
            "OnlineSecurity": Column(
                "category",
                checks=_cat_isin(INTERNET_ADDON_VALUES),
            ),
            "OnlineBackup": Column(
                "category",
                checks=_cat_isin(INTERNET_ADDON_VALUES),
            ),
            "DeviceProtection": Column(
                "category",
                checks=_cat_isin(INTERNET_ADDON_VALUES),
            ),
            "TechSupport": Column(
                "category",
                checks=_cat_isin(INTERNET_ADDON_VALUES),
            ),
            "StreamingTV": Column(
                "category",
                checks=_cat_isin(INTERNET_ADDON_VALUES),
            ),
            "StreamingMovies": Column(
                "category",
                checks=_cat_isin(INTERNET_ADDON_VALUES),
            ),
            "Contract": Column(
                "category",
                checks=_cat_isin(CONTRACT_VALUES),
                nullable=False,
            ),
            "PaperlessBilling": Column(
                "category",
                checks=_cat_isin(YES_NO),
                nullable=False,
            ),
            "PaymentMethod": Column(
                "category",
                checks=_cat_isin(PAYMENT_METHOD_VALUES),
                nullable=False,
            ),
            "MonthlyCharges": Column(
                "float32",
                checks=Check.greater_than_or_equal_to(0.0),
                nullable=False,
            ),
            "TotalCharges": Column(
                "float32",
                checks=Check.greater_than_or_equal_to(0.0),
                nullable=True,  # 11 filas vienen con cadena vacía -> NaN
                description="Suma facturada desde el alta; NaN cuando tenure == 0.",
            ),
            "Churn": Column(
                "category",
                checks=_cat_isin(YES_NO),
                nullable=False,
                description="Variable objetivo: cancelación voluntaria del servicio.",
            ),
        },
        checks=[
            Check(
                _check_phone_lines_consistency,
                name="phone_lines_consistency",
                error="Si PhoneService=No, MultipleLines debe ser 'No phone service'.",
            ),
            Check(
                _check_internet_addons_consistency,
                name="internet_addons_consistency",
                error="Si InternetService=No, los add-ons deben ser 'No internet service'.",
            ),
        ],
        strict=True,
        coerce=False,
        ordered=False,
        name="telco_customer_churn_raw",
    )


# Esquema instanciado a nivel de módulo para importación directa.
RAW_SCHEMA: Final[DataFrameSchema] = build_raw_schema()
