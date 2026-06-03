"""API REST de inferencia de ChurnLens (Fase 4).

Aplicación FastAPI que expone el modelo ganador de la Fase 3 como servicio
HTTP listo para producción:

* ``GET  /health``        — liveness/readiness (modelo cargado y operativo).
* ``GET  /metadata``      — manifiesto auditable del modelo servido.
* ``POST /predict``       — puntúa un cliente.
* ``POST /predict/batch`` — puntúa hasta 1 000 clientes por request.

Decisiones de diseño:

* El :class:`~churnlens.serving.service.ChurnScorer` se carga **una sola vez**
  en el *lifespan* de la aplicación (no por request) y se guarda en
  ``app.state.scorer``. Si los artefactos faltan, el proceso falla rápido al
  arrancar — nunca sirve tráfico a medias.
* Los endpoints de inferencia son funciones síncronas (`def`): FastAPI las
  ejecuta en su *threadpool*, de modo que el trabajo CPU-bound de NumPy /
  scikit-learn no bloquea el event loop.
* Cada respuesta incluye el header ``X-Process-Time-Ms`` para observabilidad
  de latencia sin instrumentación externa.
* La documentación interactiva (Swagger / ReDoc) se genera automáticamente en
  ``/docs`` y ``/redoc`` a partir de los contratos de
  :mod:`churnlens.serving.schemas`.

Uso local:

```bash
churnlens serve                  # equivale a uvicorn churnlens.serving.api:app
make serve                       # idem, vía Makefile
```
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from churnlens import __version__
from churnlens.logger import get_logger
from churnlens.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    CustomerPayload,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
)
from churnlens.serving.service import ChurnScorer

log = get_logger(__name__)

API_TITLE = "ChurnLens API"
API_DESCRIPTION = (
    "Servicio de inferencia para la predicción temprana de churn en servicios "
    "por suscripción. Proyecto aplicado del Módulo 6 — Diplomado MLDS · "
    "Universidad Nacional de Colombia."
)


def create_app(scorer: ChurnScorer | None = None) -> FastAPI:
    """Construye la aplicación FastAPI.

    Args:
        scorer: scorer preconstruido (inyección para tests). Si es ``None``,
            se carga desde la configuración global durante el *startup*.

    Returns:
        Aplicación FastAPI lista para ser servida por uvicorn.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Carga el scorer al arrancar (fail-fast si faltan artefactos)."""
        app.state.scorer = scorer if scorer is not None else ChurnScorer()
        log.info(
            "api_ready",
            model=app.state.scorer.model_name,
            threshold=app.state.scorer.threshold,
            version=__version__,
        )
        yield

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        contact={"name": "ChurnLens", "url": "https://github.com/jhonevergallegoate/churnlens"},
        license_info={"name": "MIT"},
    )

    @app.middleware("http")
    async def add_process_time_header(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Anota la latencia de cada request en el header ``X-Process-Time-Ms``."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # ------------------------------------------------------------------
    # Operación
    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        """Liveness/readiness: responde 200 si el modelo está cargado."""
        scorer_: ChurnScorer = request.app.state.scorer
        return HealthResponse(status="ok", model=scorer_.model_name, version=__version__)

    @app.get("/metadata", response_model=ModelMetadataResponse, tags=["ops"])
    def metadata(request: Request) -> ModelMetadataResponse:
        """Manifiesto auditable del modelo servido (algoritmo, métricas, hash)."""
        scorer_: ChurnScorer = request.app.state.scorer
        meta = scorer_.metadata
        metrics_val_tuned = (meta.get("metrics") or {}).get("val_tuned") or {}
        return ModelMetadataResponse(
            model=scorer_.model_name,
            algorithm=str(meta.get("algorithm", "?")),
            version=__version__,
            created_at=str(meta.get("created_at", "?")),
            threshold=scorer_.threshold,
            n_features=len(scorer_.feature_set),
            metrics_val={
                k: float(v) for k, v in metrics_val_tuned.items() if isinstance(v, (int, float))
            },
            hash_model=str(meta.get("hash_model", "?")),
        )

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------
    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(payload: CustomerPayload, request: Request) -> PredictionResponse:
        """Puntúa un cliente y devuelve probabilidad, decisión y banda de riesgo."""
        scorer_: ChurnScorer = request.app.state.scorer
        prediction = scorer_.predict_payloads([payload])[0]
        return PredictionResponse(
            **prediction.model_dump(),
            model=scorer_.model_name,
            threshold=scorer_.threshold,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    def predict_batch(batch: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
        """Puntúa un lote de clientes (máx. 1 000) preservando el orden del request."""
        scorer_: ChurnScorer = request.app.state.scorer
        predictions = scorer_.predict_payloads(batch.customers)
        n_churn = sum(p.prediction for p in predictions)
        n_total = len(predictions)
        return BatchPredictionResponse(
            model=scorer_.model_name,
            threshold=scorer_.threshold,
            summary=BatchSummary(
                n_customers=n_total,
                n_predicted_churn=n_churn,
                churn_rate=round(n_churn / n_total, 6),
                mean_probability=round(sum(p.probability for p in predictions) / n_total, 6),
            ),
            predictions=predictions,
        )

    return app


# Instancia module-level para `uvicorn churnlens.serving.api:app`.
# La carga del modelo ocurre en el lifespan (startup), no en el import.
app = create_app()
