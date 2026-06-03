"""Servicio de inferencia: encapsula el pipeline completo de scoring.

:class:`ChurnScorer` reconstruye en producción exactamente el mismo camino
que se usó en entrenamiento (Fases 2–3):

```
payload crudo (19 columnas del contrato)
    → add_engineered_features()        (Fase 2 — features derivadas)
    → ColumnTransformer.transform()    (Fase 2 — preprocessor.joblib)
    → reordenar al feature_set         (manifiesto del modelo)
    → model.predict_proba()[:, 1]      (Fase 3 — logreg_l1.joblib)
    → decisión con threshold sintonizado (0.58 por defecto)
```

Los tres artefactos (modelo, manifiesto y preprocesador) se cargan **una sola
vez** al construir el scorer; cada request solo paga el costo de `transform`
+ `predict_proba`, lo que mantiene la latencia en milisegundos.

El módulo no importa `churnlens.models.train` a propósito: ese módulo arrastra
LightGBM y otras dependencias de entrenamiento que no se necesitan (ni deben
pesar) en la imagen de producción.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from churnlens.config import Settings
from churnlens.config import settings as default_settings
from churnlens.features.engineering import add_engineered_features
from churnlens.logger import get_logger
from churnlens.models.registry import load_model
from churnlens.serving.schemas import CustomerPayload, CustomerPrediction, RiskBand

log = get_logger(__name__)

DEFAULT_THRESHOLD: float = 0.5
"""Fallback si el manifiesto del modelo no registra un threshold sintonizado."""

PREPROCESSOR_FILENAME: str = "preprocessor.joblib"
"""Nombre canónico del transformador ajustado en Fase 2 (en `data/processed/`)."""


def _resolve_threshold(metadata: dict[str, Any], override: float | None) -> float:
    """Devuelve el threshold operativo: override explícito o el sintonizado en Fase 3."""
    if override is not None:
        return float(override)
    metrics = metadata.get("metrics") or {}
    tuned = metrics.get("val_tuned") or {}
    return float(tuned.get("threshold", DEFAULT_THRESHOLD))


class ChurnScorer:
    """Pipeline de inferencia listo para producción.

    Args:
        settings: configuración del proyecto (default: instancia global).
        model_name: nombre lógico del modelo en el registro
            (default: ``settings.serving_model_name``).
        models_dir: directorio del registro de modelos
            (default: ``settings.models_dir``).
        preprocessor_path: ruta del `ColumnTransformer` ajustado
            (default: ``settings.processed_dir / "preprocessor.joblib"``).
        threshold: threshold de decisión; si es ``None`` se usa
            ``settings.serving_threshold`` y, en su defecto, el threshold
            sintonizado registrado en el manifiesto del modelo.

    Raises:
        FileNotFoundError: si falta el modelo, el manifiesto o el preprocesador.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_name: str | None = None,
        models_dir: Path | str | None = None,
        preprocessor_path: Path | str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Carga modelo + manifiesto + preprocesador y resuelve el threshold."""
        cfg = settings or default_settings
        self.model_name: str = model_name or cfg.serving_model_name
        self.model, self.metadata = load_model(
            self.model_name,
            models_dir=models_dir if models_dir is not None else cfg.models_dir,
        )

        pre_path = (
            Path(preprocessor_path)
            if preprocessor_path is not None
            else cfg.processed_dir / PREPROCESSOR_FILENAME
        )
        if not pre_path.exists():
            msg = (
                f"Preprocesador no encontrado en {pre_path}. "
                "Ejecuta primero `make preprocess` (Fase 2)."
            )
            raise FileNotFoundError(msg)
        self.preprocessor: Any = joblib.load(pre_path)
        self.preprocessor_path: Path = pre_path

        self.feature_set: list[str] = list(self.metadata.get("feature_set") or [])
        self.input_columns: list[str] = [
            str(c) for c in getattr(self.preprocessor, "feature_names_in_", [])
        ]
        self.threshold: float = _resolve_threshold(
            self.metadata,
            threshold if threshold is not None else cfg.serving_threshold,
        )

        log.info(
            "scorer_loaded",
            model=self.model_name,
            algorithm=str(self.metadata.get("algorithm", "?")),
            threshold=self.threshold,
            n_features=len(self.feature_set),
            preprocessor=str(pre_path),
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def predict_proba_frame(self, df_raw: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Puntúa un DataFrame con el contrato crudo (sin ``Churn``).

        Args:
            df_raw: una fila por cliente, con las columnas originales del
                dataset (los tipos numéricos ya casteados).

        Returns:
            Vector de probabilidades P(churn) en el mismo orden de filas.

        Raises:
            ValueError: si faltan columnas requeridas por el preprocesador.
        """
        df = add_engineered_features(df_raw)
        if self.input_columns:
            missing = [c for c in self.input_columns if c not in df.columns]
            if missing:
                msg = f"Faltan columnas requeridas por el preprocesador: {missing}"
                raise ValueError(msg)
            df = df[self.input_columns]

        transformed = np.asarray(self.preprocessor.transform(df))
        feature_names = [str(c) for c in self.preprocessor.get_feature_names_out()]
        frame = pd.DataFrame(transformed, columns=feature_names)
        if self.feature_set:
            frame = frame[self.feature_set]

        proba = self.model.predict_proba(frame.astype("float32").to_numpy())[:, 1]
        return np.asarray(proba, dtype="float64")

    def predict_payloads(self, payloads: Sequence[CustomerPayload]) -> list[CustomerPrediction]:
        """Puntúa una secuencia de payloads validados y arma las predicciones.

        Args:
            payloads: clientes ya validados por Pydantic (contrato de la API).

        Returns:
            Una :class:`CustomerPrediction` por payload, en el mismo orden.
        """
        df_raw = self._payloads_to_frame(payloads)
        proba = self.predict_proba_frame(df_raw)
        return [
            self._to_prediction(payload, float(p))
            for payload, p in zip(payloads, proba, strict=True)
        ]

    def risk_band(self, probability: float) -> RiskBand:
        """Asigna la banda de riesgo comercial a partir de la probabilidad.

        Reglas (relativas al threshold vigente ``t``):
        ``high`` si p ≥ t · ``medium`` si p ≥ t/2 · ``low`` en el resto.
        """
        if probability >= self.threshold:
            return "high"
        if probability >= self.threshold / 2:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    @staticmethod
    def _payloads_to_frame(payloads: Sequence[CustomerPayload]) -> pd.DataFrame:
        """Convierte payloads Pydantic en un DataFrame con los dtypes del contrato."""
        records = [p.model_dump(exclude={"customerID"}) for p in payloads]
        df = pd.DataFrame.from_records(records)
        df["tenure"] = df["tenure"].astype("int16")
        df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")
        df["MonthlyCharges"] = df["MonthlyCharges"].astype("float32")
        # None → NaN; la mediana del entrenamiento la imputa el preprocesador.
        df["TotalCharges"] = df["TotalCharges"].astype("float32")
        return df

    def _to_prediction(self, payload: CustomerPayload, probability: float) -> CustomerPrediction:
        """Arma la predicción final de un cliente a partir de su probabilidad."""
        is_churn = probability >= self.threshold
        return CustomerPrediction(
            customerID=payload.customerID,
            probability=round(probability, 6),
            prediction=int(is_churn),
            label="churn" if is_churn else "no_churn",
            risk_band=self.risk_band(probability),
        )
