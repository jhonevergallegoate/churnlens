"""Punto de entrada oficial de la Fase 4 (TDSP · despliegue).

Implementa la verificación end-to-end del entregable **"Código de
despliegue"** exigido por la rúbrica del Módulo 6 del Diplomado MLDS.
El script:

1. Verifica que los artefactos de inferencia existan (modelo registrado +
   preprocesador de Fase 2). Con ``--ensure-artifacts`` los reconstruye
   si faltan (descarga → preprocesamiento → entrenamiento del ganador).
2. Levanta la API FastAPI **in-process** (TestClient — sin abrir puertos)
   y ejercita los cuatro endpoints: ``/health``, ``/metadata``,
   ``/predict`` y ``/predict/batch``.
3. Valida los contratos de respuesta (status codes, rangos de probabilidad,
   consistencia del summary) y mide la latencia de cada endpoint.
4. Persiste la evidencia en ``reports/tables/deployment_smoke.json``.

El smoke real sobre HTTP/Docker se hace en CI (job ``docker-smoke``) y
con ``make docker-up``.

Uso:

```bash
python scripts/deployment/main.py [--ensure-artifacts] [--quiet]
make deploy-smoke
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from churnlens.config import Settings
from churnlens.logger import get_logger
from churnlens.serving.service import PREPROCESSOR_FILENAME

log = get_logger("scripts.deployment")

# Clientes de muestra: cubren los tres niveles de riesgo y el caso borde
# tenure=0 con TotalCharges nulo (cliente recién dado de alta).
SAMPLE_HIGH_RISK: dict[str, Any] = {
    "customerID": "9237-HQITU",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65,
}

SAMPLE_LOW_RISK: dict[str, Any] = {
    "customerID": "7795-CFOCW",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 45,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 42.30,
    "TotalCharges": 1840.75,
}

SAMPLE_NEW_CUSTOMER: dict[str, Any] = {
    "customerID": "4472-LVYGI",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 0,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "No",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 52.55,
    "TotalCharges": None,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deployment",
        description="Smoke test end-to-end de la API de inferencia (Fase 4).",
    )
    parser.add_argument(
        "--ensure-artifacts",
        action="store_true",
        help="Reconstruye los artefactos (datos + preprocesador + modelo) si faltan.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity a WARNING.",
    )
    return parser.parse_args(argv)


def _artifacts_missing(settings: Settings) -> list[Path]:
    """Devuelve la lista de artefactos de inferencia que faltan."""
    required = [
        settings.models_dir / f"{settings.serving_model_name}.joblib",
        settings.models_dir / f"{settings.serving_model_name}.metadata.json",
        settings.processed_dir / PREPROCESSOR_FILENAME,
    ]
    return [p for p in required if not p.exists()]


def _ensure_artifacts(settings: Settings) -> None:
    """Reconstruye los artefactos mínimos para servir el modelo."""
    from churnlens.data.loader import TelcoChurnLoader
    from churnlens.features.pipeline import run_preprocessing
    from churnlens.models.train import train_models

    log.info("deployment_rebuilding_artifacts")
    TelcoChurnLoader(settings=settings).download()
    run_preprocessing(settings=settings)
    train_models(models=[settings.serving_model_name], cv=3)
    log.info("deployment_artifacts_ready")


def _exercise_api(settings: Settings) -> dict[str, Any]:
    """Levanta la API in-process y ejercita los cuatro endpoints del contrato."""
    # Import tardío: requiere fastapi/httpx ya instalados (extras dev).
    from fastapi.testclient import TestClient

    from churnlens.serving.api import create_app

    report: dict[str, Any] = {"checks": []}

    def _check(name: str, ok: bool, detail: dict[str, Any]) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), **detail})
        if not ok:
            log.error("deployment_check_failed", check=name, **detail)

    with TestClient(create_app()) as client:
        # ---- /health -------------------------------------------------
        t0 = time.perf_counter()
        resp = client.get("/health")
        health_ms = (time.perf_counter() - t0) * 1000.0
        body = resp.json()
        _check(
            "GET /health",
            resp.status_code == 200 and body.get("status") == "ok",
            {"status_code": resp.status_code, "latency_ms": round(health_ms, 2), "body": body},
        )

        # ---- /metadata -----------------------------------------------
        resp = client.get("/metadata")
        meta = resp.json()
        _check(
            "GET /metadata",
            resp.status_code == 200
            and meta.get("model") == settings.serving_model_name
            and 0.0 < float(meta.get("threshold", 0)) < 1.0
            and int(meta.get("n_features", 0)) > 0,
            {"status_code": resp.status_code, "body": meta},
        )

        # ---- /predict (alto riesgo) ------------------------------------
        t0 = time.perf_counter()
        resp = client.post("/predict", json=SAMPLE_HIGH_RISK)
        predict_ms = (time.perf_counter() - t0) * 1000.0
        pred = resp.json()
        _check(
            "POST /predict",
            resp.status_code == 200
            and 0.0 <= pred.get("probability", -1) <= 1.0
            and pred.get("prediction") in (0, 1)
            and pred.get("customerID") == SAMPLE_HIGH_RISK["customerID"],
            {"status_code": resp.status_code, "latency_ms": round(predict_ms, 2), "body": pred},
        )

        # ---- /predict (payload inválido → 422) -------------------------
        invalid = dict(SAMPLE_HIGH_RISK, Contract="Three year")
        resp = client.post("/predict", json=invalid)
        _check(
            "POST /predict (payload inválido)",
            resp.status_code == 422,
            {"status_code": resp.status_code},
        )

        # ---- /predict/batch --------------------------------------------
        batch = {"customers": [SAMPLE_HIGH_RISK, SAMPLE_LOW_RISK, SAMPLE_NEW_CUSTOMER]}
        t0 = time.perf_counter()
        resp = client.post("/predict/batch", json=batch)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        body = resp.json()
        summary = body.get("summary", {})
        predictions = body.get("predictions", [])
        _check(
            "POST /predict/batch",
            resp.status_code == 200
            and summary.get("n_customers") == 3
            and len(predictions) == 3
            and summary.get("n_predicted_churn") == sum(p.get("prediction", 0) for p in predictions)
            and all(0.0 <= p.get("probability", -1) <= 1.0 for p in predictions),
            {"status_code": resp.status_code, "latency_ms": round(batch_ms, 2), "body": body},
        )

    report["model"] = settings.serving_model_name
    report["n_checks"] = len(report["checks"])
    report["n_failed"] = sum(1 for c in report["checks"] if not c["ok"])
    report["ok"] = report["n_failed"] == 0
    return report


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del smoke test de despliegue."""
    args = _parse_args(argv)
    settings = Settings(log_level="WARNING") if args.quiet else Settings()

    missing = _artifacts_missing(settings)
    if missing and args.ensure_artifacts:
        _ensure_artifacts(settings)
        missing = _artifacts_missing(settings)
    if missing:
        log.error(
            "deployment_artifacts_missing",
            missing=[str(p) for p in missing],
            hint="Ejecuta `make phase3` o usa --ensure-artifacts.",
        )
        return 1

    report = _exercise_api(settings)

    tables_dir = settings.project_root / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "deployment_smoke.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("deployment_smoke_done", ok=report["ok"], report=str(out_path))

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
