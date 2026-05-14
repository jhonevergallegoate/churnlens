"""Fixtures globales del proyecto."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

# CSV mínimo con TODAS las columnas y combinaciones válidas del dataset Telco.
# Sirve para probar el loader sin depender de red.
_RAW_CSV_TEMPLATE = """\
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.30,1840.75,No
9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,151.65,Yes
4472-LVYGI,Female,0,Yes,Yes,0,No,No phone service,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Two year,No,Bank transfer (automatic),52.55, ,No
"""


@pytest.fixture
def sample_csv_text() -> str:
    """Devuelve el contenido textual de un CSV de muestra válido."""
    return _RAW_CSV_TEMPLATE


@pytest.fixture
def sample_dataframe(sample_csv_text: str) -> pd.DataFrame:
    """Devuelve el CSV de muestra ya leído como DataFrame `object`."""
    return pd.read_csv(StringIO(sample_csv_text), dtype="object")
