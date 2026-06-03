"""Tests de la API REST de inferencia (`churnlens.serving.api`)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from churnlens import __version__
from churnlens.serving.api import create_app
from churnlens.serving.service import ChurnScorer

from .test_serving_service import BASE_PAYLOAD


@pytest.fixture(scope="module")
def client(serving_scorer: ChurnScorer) -> Iterator[TestClient]:
    """TestClient con el scorer de sesión inyectado (lifespan incluido)."""
    with TestClient(create_app(scorer=serving_scorer)) as test_client:
        yield test_client


def _payload(**overrides: Any) -> dict[str, Any]:
    return {**BASE_PAYLOAD, **overrides}


class TestOpsEndpoints:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "logreg_test"
        assert body["version"] == __version__

    def test_metadata_manifest(self, client: TestClient) -> None:
        resp = client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "logreg_test"
        assert body["algorithm"] == "LogisticRegression"
        assert body["threshold"] == pytest.approx(0.58)
        assert body["n_features"] > 0
        assert "pr_auc" in body["metrics_val"]
        assert len(body["hash_model"]) == 64  # SHA-256 hex

    def test_latency_header_present(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "X-Process-Time-Ms" in resp.headers
        assert float(resp.headers["X-Process-Time-Ms"]) >= 0.0

    def test_openapi_docs_available(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert {"/health", "/metadata", "/predict", "/predict/batch"} <= set(paths)


class TestPredictEndpoint:
    def test_predict_returns_valid_prediction(self, client: TestClient) -> None:
        resp = client.post("/predict", json=_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["prediction"] in (0, 1)
        assert body["label"] in ("churn", "no_churn")
        assert body["risk_band"] in ("low", "medium", "high")
        assert body["customerID"] == BASE_PAYLOAD["customerID"]
        assert body["model"] == "logreg_test"
        assert body["threshold"] == pytest.approx(0.58)

    def test_invalid_domain_rejected_422(self, client: TestClient) -> None:
        resp = client.post("/predict", json=_payload(Contract="Three year"))
        assert resp.status_code == 422

    def test_cross_field_inconsistency_rejected_422(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            json=_payload(PhoneService="No", MultipleLines="Yes"),
        )
        assert resp.status_code == 422

    def test_extra_field_rejected_422(self, client: TestClient) -> None:
        resp = client.post("/predict", json=_payload(Churn="Yes"))
        assert resp.status_code == 422

    def test_missing_required_field_rejected_422(self, client: TestClient) -> None:
        payload = _payload()
        payload.pop("Contract")
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch_summary_consistent(self, client: TestClient) -> None:
        customers = [
            _payload(customerID="AAAA-00001"),
            _payload(customerID="BBBB-00002", tenure=60, Contract="Two year"),
            _payload(customerID="CCCC-00003", tenure=12),
        ]
        resp = client.post("/predict/batch", json={"customers": customers})
        assert resp.status_code == 200
        body = resp.json()
        predictions = body["predictions"]
        summary = body["summary"]

        assert summary["n_customers"] == 3
        assert len(predictions) == 3
        assert [p["customerID"] for p in predictions] == [
            "AAAA-00001",
            "BBBB-00002",
            "CCCC-00003",
        ]
        assert summary["n_predicted_churn"] == sum(p["prediction"] for p in predictions)
        mean_proba = sum(p["probability"] for p in predictions) / 3
        assert summary["mean_probability"] == pytest.approx(mean_proba, abs=1e-4)

    def test_empty_batch_rejected_422(self, client: TestClient) -> None:
        resp = client.post("/predict/batch", json={"customers": []})
        assert resp.status_code == 422

    def test_single_invalid_customer_rejects_batch_422(self, client: TestClient) -> None:
        customers = [_payload(), _payload(tenure=999)]
        resp = client.post("/predict/batch", json={"customers": customers})
        assert resp.status_code == 422
