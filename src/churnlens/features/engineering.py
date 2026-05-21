"""Ingeniería de variables derivadas para el dataset Telco Customer Churn.

Las funciones de este módulo operan sobre un DataFrame que **ya pasó** por el
loader oficial (`churnlens.data.loader.TelcoChurnLoader.load_validated`),
por lo que se asume que:

* Los tipos están casteados según `RAW_SCHEMA`.
* Las restricciones de integridad cruzada se cumplen (p. ej. ``PhoneService=No``
  implica ``MultipleLines="No phone service"``).

La función pública principal es `add_engineered_features`, que añade un
bloque de columnas nuevas sin alterar las originales. Las features se diseñaron
con dos criterios:

1. **Interpretabilidad** — cada feature mapea a una hipótesis de negocio
   sobre el comportamiento de churn.
2. **No leakage** — solo se usan columnas observables antes de la cancelación.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes públicas del esquema derivado
# ---------------------------------------------------------------------------
ADDON_COLS: Final[tuple[str, ...]] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
"""Add-ons de Internet — cada uno contribuye 1 a `services_count` cuando es 'Yes'."""

SERVICE_COLS: Final[tuple[str, ...]] = (
    "PhoneService",
    "MultipleLines",
    *ADDON_COLS,
)
"""Conjunto completo de servicios contables hacia `services_count`."""

TENURE_BUCKET_EDGES: Final[tuple[int, ...]] = (-1, 12, 24, 48, 72)
"""Bordes de los buckets de antigüedad (en meses). El -1 permite incluir tenure=0."""

TENURE_BUCKET_LABELS: Final[tuple[str, ...]] = (
    "0-12m",
    "13-24m",
    "25-48m",
    "49-72m",
)
"""Etiquetas humanas de los buckets de antigüedad."""

_AUTOMATIC_METHODS: Final[frozenset[str]] = frozenset(
    {"Bank transfer (automatic)", "Credit card (automatic)"}
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del DataFrame con _features_ derivadas añadidas.

    Las columnas originales no se alteran. Las nuevas columnas son:

    * ``tenure_bucket`` (`category` ordinal): partición de la antigüedad en
      cuatro tramos calibrados con la rúbrica del dataset.
    * ``services_count`` (`int8`): número total de servicios y add-ons
      contratados — sirve como _proxy_ de "lock-in".
    * ``has_internet`` (`bool`): True si el cliente tiene cualquier tipo de
      servicio de internet.
    * ``has_phone`` (`bool`): True si el cliente tiene servicio telefónico.
    * ``auto_payment`` (`bool`): True si el método de pago es automático
      (transferencia o tarjeta) — _proxy_ de fricción de cancelación.
    * ``avg_monthly_spend`` (`float32`): gasto promedio observado por mes
      (``TotalCharges / max(tenure, 1)``). Útil para detectar cambios de
      plan.
    * ``monthly_spend_gap`` (`float32`): diferencia entre el cargo mensual
      actual y el promedio histórico — captura _upsell / downsell_.

    Args:
        df: DataFrame validado producido por el _loader_ oficial.

    Returns:
        DataFrame con las nuevas columnas añadidas a la derecha.

    Raises:
        KeyError: si falta alguna columna esperada del esquema crudo.
    """
    required = {
        "tenure",
        "PhoneService",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        *SERVICE_COLS,
    }
    missing = required.difference(df.columns)
    if missing:
        msg = f"Faltan columnas requeridas para feature engineering: {sorted(missing)}"
        raise KeyError(msg)

    out = df.copy()

    out["tenure_bucket"] = pd.cut(
        out["tenure"].astype("int16"),
        bins=list(TENURE_BUCKET_EDGES),
        labels=list(TENURE_BUCKET_LABELS),
        include_lowest=True,
        ordered=True,
    ).astype(pd.CategoricalDtype(categories=list(TENURE_BUCKET_LABELS), ordered=True))

    out["services_count"] = _count_services(out).astype("int8")
    out["has_internet"] = (out["InternetService"].astype(str) != "No").astype(bool)
    out["has_phone"] = (out["PhoneService"].astype(str) == "Yes").astype(bool)
    out["auto_payment"] = out["PaymentMethod"].astype(str).isin(_AUTOMATIC_METHODS).astype(bool)

    safe_tenure = out["tenure"].clip(lower=1).astype("float32")
    out["avg_monthly_spend"] = (
        out["TotalCharges"].astype("float32").fillna(0.0) / safe_tenure
    ).astype("float32")
    out["monthly_spend_gap"] = (
        out["MonthlyCharges"].astype("float32") - out["avg_monthly_spend"]
    ).astype("float32")

    return out


def _count_services(df: pd.DataFrame) -> pd.Series:
    """Cuenta el número de servicios y add-ons activos por cliente."""
    counts = np.zeros(len(df), dtype="int8")
    for col in SERVICE_COLS:
        counts = counts + (df[col].astype(str) == "Yes").astype("int8").to_numpy()
    return pd.Series(counts, index=df.index, name="services_count")
