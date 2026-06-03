"""Contratos de entrada/salida de la API de inferencia (Fase 4).

El contrato de entrada (:class:`CustomerPayload`) replica columna por columna
el esquema crudo del dataset (`churnlens.data.schema.RAW_SCHEMA`), excluyendo
la variable objetivo ``Churn``. Los nombres de campo conservan el *casing*
original del dataset (``SeniorCitizen``, ``MonthlyCharges``, …) para que el
payload de la API sea un espejo exacto del contrato de datos documentado en
`docs/data/data_dictionary.md` — por eso este módulo tiene una excepción de
naming (N815) declarada en ``pyproject.toml``.

Además de los dominios por campo, se replican como validadores Pydantic las
dos reglas de integridad cruzada del esquema Pandera:

1. ``PhoneService = "No"``  ⟹  ``MultipleLines = "No phone service"``.
2. ``InternetService = "No"``  ⟹  todos los add-ons = ``"No internet service"``.

Esto garantiza que ninguna observación fuera de contrato llegue al modelo:
la API responde ``422 Unprocessable Entity`` con el detalle del error.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Alias de dominios (espejo de churnlens.data.schema)
# ---------------------------------------------------------------------------
YesNo = Literal["Yes", "No"]
InternetAddon = Literal["No internet service", "No", "Yes"]
RiskBand = Literal["low", "medium", "high"]
"""Banda de riesgo: ``high`` ≥ threshold; ``medium`` ≥ threshold/2; ``low`` el resto."""

MAX_BATCH_SIZE: int = 1_000
"""Límite de clientes por request de batch — protege la latencia del servicio."""


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
class CustomerPayload(BaseModel):
    """Un cliente a puntuar — espejo del contrato crudo del dataset (sin ``Churn``)."""

    model_config = ConfigDict(extra="forbid")

    customerID: str | None = Field(
        default=None,
        description="Identificador del cliente (opcional, solo trazabilidad — no entra al modelo).",
    )
    gender: Literal["Female", "Male"] = Field(description="Género auto-reportado del titular.")
    SeniorCitizen: Literal[0, 1] = Field(
        default=0,
        description="1 si el cliente es senior (65+). No es usada por el modelo actual.",
    )
    Partner: YesNo = Field(description="Si el cliente convive con pareja.")
    Dependents: YesNo = Field(description="Si el cliente tiene dependientes.")
    tenure: int = Field(ge=0, le=72, description="Antigüedad en meses desde el alta (0–72).")
    PhoneService: YesNo = Field(description="Si tiene servicio telefónico.")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(
        description="Si tiene múltiples líneas telefónicas."
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        description="Tipo de servicio de internet contratado."
    )
    OnlineSecurity: InternetAddon = Field(description="Add-on de seguridad en línea.")
    OnlineBackup: InternetAddon = Field(description="Add-on de respaldo en línea.")
    DeviceProtection: InternetAddon = Field(description="Add-on de protección de dispositivos.")
    TechSupport: InternetAddon = Field(description="Add-on de soporte técnico.")
    StreamingTV: InternetAddon = Field(description="Add-on de TV en streaming.")
    StreamingMovies: InternetAddon = Field(description="Add-on de películas en streaming.")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="Tipo de contrato vigente."
    )
    PaperlessBilling: YesNo = Field(description="Si la facturación es electrónica.")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(description="Método de pago registrado.")
    MonthlyCharges: float = Field(ge=0, description="Cargo mensual vigente (USD).")
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        description="Total facturado histórico (USD). Null cuando tenure = 0 (se imputa).",
    )

    @model_validator(mode="after")
    def _check_cross_field_consistency(self) -> CustomerPayload:
        """Replica las reglas de integridad cruzada del esquema Pandera."""
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            msg = "Si PhoneService='No', MultipleLines debe ser 'No phone service'."
            raise ValueError(msg)
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            msg = "Si PhoneService='Yes', MultipleLines no puede ser 'No phone service'."
            raise ValueError(msg)

        addons: dict[str, str] = {
            "OnlineSecurity": self.OnlineSecurity,
            "OnlineBackup": self.OnlineBackup,
            "DeviceProtection": self.DeviceProtection,
            "TechSupport": self.TechSupport,
            "StreamingTV": self.StreamingTV,
            "StreamingMovies": self.StreamingMovies,
        }
        if self.InternetService == "No":
            inconsistent = [k for k, v in addons.items() if v != "No internet service"]
            if inconsistent:
                msg = (
                    "Si InternetService='No', estos add-ons deben ser "
                    f"'No internet service': {inconsistent}"
                )
                raise ValueError(msg)
        else:
            inconsistent = [k for k, v in addons.items() if v == "No internet service"]
            if inconsistent:
                msg = (
                    "Con internet activo, estos add-ons no pueden ser "
                    f"'No internet service': {inconsistent}"
                )
                raise ValueError(msg)
        return self


class BatchPredictionRequest(BaseModel):
    """Lote de clientes a puntuar en una sola llamada."""

    customers: list[CustomerPayload] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Entre 1 y {MAX_BATCH_SIZE} clientes por request.",
    )


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------
class CustomerPrediction(BaseModel):
    """Resultado de inferencia para un cliente."""

    customerID: str | None = Field(default=None, description="Eco del identificador recibido.")
    probability: float = Field(ge=0, le=1, description="P(churn) estimada por el modelo.")
    prediction: int = Field(
        ge=0, le=1, description="1 si P(churn) ≥ threshold, 0 en caso contrario."
    )
    label: Literal["churn", "no_churn"] = Field(description="Etiqueta legible de la decisión.")
    risk_band: RiskBand = Field(description="Banda de riesgo para priorización comercial.")


class PredictionResponse(CustomerPrediction):
    """Respuesta de ``POST /predict`` — predicción + contexto del modelo."""

    model: str = Field(description="Nombre del modelo registrado que generó la predicción.")
    threshold: float = Field(description="Threshold de decisión vigente.")


class BatchSummary(BaseModel):
    """Agregados del lote puntuado."""

    n_customers: int = Field(description="Clientes recibidos en el lote.")
    n_predicted_churn: int = Field(description="Clientes con predicción de churn (=1).")
    churn_rate: float = Field(ge=0, le=1, description="Proporción de churn predicho en el lote.")
    mean_probability: float = Field(ge=0, le=1, description="P(churn) promedio del lote.")


class BatchPredictionResponse(BaseModel):
    """Respuesta de ``POST /predict/batch``."""

    model: str = Field(description="Nombre del modelo registrado que generó las predicciones.")
    threshold: float = Field(description="Threshold de decisión vigente.")
    summary: BatchSummary
    predictions: list[CustomerPrediction] = Field(
        description="Una predicción por cliente, en el mismo orden del request."
    )


class HealthResponse(BaseModel):
    """Respuesta de ``GET /health`` — liveness/readiness del servicio."""

    status: Literal["ok"] = Field(description="'ok' si el modelo está cargado y operativo.")
    model: str = Field(description="Nombre del modelo servido.")
    version: str = Field(description="Versión del paquete churnlens.")


class ModelMetadataResponse(BaseModel):
    """Respuesta de ``GET /metadata`` — manifiesto auditable del modelo servido."""

    model: str = Field(description="Nombre lógico del modelo en el registro.")
    algorithm: str = Field(description="Clase del estimador (p. ej. LogisticRegression).")
    version: str = Field(description="Versión del paquete churnlens.")
    created_at: str = Field(description="Fecha de entrenamiento (UTC ISO-8601).")
    threshold: float = Field(description="Threshold de decisión vigente.")
    n_features: int = Field(description="Número de features de entrada al estimador.")
    metrics_val: dict[str, float] = Field(
        description="Métricas en validación al threshold sintonizado (Fase 3)."
    )
    hash_model: str = Field(description="SHA-256 del artefacto joblib (anti-tampering).")
