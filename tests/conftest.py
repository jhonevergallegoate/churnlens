"""Fixtures globales del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from churnlens.data.loader import TelcoChurnLoader

if TYPE_CHECKING:
    from churnlens.serving.service import ChurnScorer

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


@pytest.fixture
def validated_sample(sample_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la muestra ya casteada al esquema oficial (post-coerce)."""
    return TelcoChurnLoader._coerce_types(sample_dataframe)


def make_synthetic_raw(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame sintético conforme al esquema crudo (dtype object).

    Reutilizable fuera de fixtures (p. ej. fixtures de sesión de serving).
    """
    rng = np.random.default_rng(seed)
    customer_id = [f"{1000 + i:04d}-ABCDE" for i in range(n)]
    gender = rng.choice(["Female", "Male"], size=n)
    senior = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    partner = rng.choice(["Yes", "No"], size=n)
    dependents = rng.choice(["Yes", "No"], size=n)
    tenure = rng.integers(0, 73, size=n)

    has_phone = rng.choice(["Yes", "No"], size=n, p=[0.9, 0.1])
    multiple_lines = np.where(
        has_phone == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n),
    )

    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20])
    addon_cols = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    addons = {
        c: np.where(
            internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n),
        )
        for c in addon_cols
    }

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    paperless = rng.choice(["Yes", "No"], size=n)
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
    )

    monthly = rng.uniform(18.25, 118.75, size=n).round(2)
    total = (monthly * np.maximum(tenure, 1)).astype(float)
    # Reproducimos las cadenas vacías para tenure=0.
    total_str = [f"{v:.2f}" if t > 0 else " " for v, t in zip(total, tenure, strict=False)]

    churn = rng.choice(["Yes", "No"], size=n, p=[0.27, 0.73])

    data = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior.astype(str),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure.astype(str),
        "PhoneService": has_phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": [f"{v:.2f}" for v in monthly],
        "TotalCharges": total_str,
        "Churn": churn,
    }
    return pd.DataFrame(data, dtype="object")


@pytest.fixture
def synthetic_churn_dataset() -> pd.DataFrame:
    """Genera un DataFrame sintético de 200 filas conforme al esquema crudo.

    Construido en `object` dtype para reflejar el formato post-`read_csv`;
    deja que el _loader_ aplique el casteo canónico.
    """
    return make_synthetic_raw()


@pytest.fixture
def validated_synthetic(synthetic_churn_dataset: pd.DataFrame) -> pd.DataFrame:
    """Synthetic dataset casteado al esquema canónico."""
    return TelcoChurnLoader._coerce_types(synthetic_churn_dataset)


# ---------------------------------------------------------------------------
# Fixtures de serving (Fase 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServingArtifacts:
    """Rutas de los artefactos mínimos para construir un ChurnScorer."""

    models_dir: Path
    model_name: str
    preprocessor_path: Path


@pytest.fixture(scope="session")
def serving_artifacts(tmp_path_factory: pytest.TempPathFactory) -> ServingArtifacts:
    """Entrena un modelo pequeño sobre datos sintéticos y persiste los artefactos.

    Replica el pipeline real (engineered features → ColumnTransformer →
    LogisticRegression → registro joblib + manifiesto) en un directorio
    temporal de sesión, de modo que los tests de serving no dependan de
    artefactos locales no versionados.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression

    from churnlens.features.engineering import add_engineered_features
    from churnlens.features.preprocessing import binarize_target, build_preprocessor
    from churnlens.models.registry import save_model

    tmp = tmp_path_factory.mktemp("serving")
    raw = TelcoChurnLoader._coerce_types(make_synthetic_raw(n=400))
    df = add_engineered_features(raw).drop(columns=["customerID"])
    y = binarize_target(df.pop("Churn"))

    preprocessor = build_preprocessor()
    x = preprocessor.fit_transform(df)
    feature_names = list(preprocessor.get_feature_names_out())

    model = LogisticRegression(max_iter=500, random_state=42).fit(x, y.to_numpy())

    models_dir = tmp / "models"
    save_model(
        model,
        "logreg_test",
        metadata={
            "feature_set": feature_names,
            "metrics": {"val_tuned": {"threshold": 0.58, "pr_auc": 0.6}},
        },
        models_dir=models_dir,
    )
    preprocessor_path = tmp / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    return ServingArtifacts(
        models_dir=models_dir,
        model_name="logreg_test",
        preprocessor_path=preprocessor_path,
    )


@pytest.fixture(scope="session")
def serving_scorer(serving_artifacts: ServingArtifacts) -> ChurnScorer:
    """ChurnScorer listo para puntuar, construido sobre los artefactos de sesión."""
    from churnlens.serving.service import ChurnScorer

    return ChurnScorer(
        model_name=serving_artifacts.model_name,
        models_dir=serving_artifacts.models_dir,
        preprocessor_path=serving_artifacts.preprocessor_path,
    )
