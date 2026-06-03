"""Tests del pipeline de inferencia (`churnlens.serving.service`)."""

from __future__ import annotations

from typing import Any

import pytest

from churnlens.serving.schemas import CustomerPayload
from churnlens.serving.service import ChurnScorer

from .conftest import ServingArtifacts

# Payload base válido — cliente de alto riesgo según la EDA de Fase 2.
BASE_PAYLOAD: dict[str, Any] = {
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


def make_payload(**overrides: Any) -> CustomerPayload:
    return CustomerPayload(**{**BASE_PAYLOAD, **overrides})


class TestScorerLoading:
    def test_loads_artifacts_and_threshold(self, serving_scorer: ChurnScorer) -> None:
        assert serving_scorer.model_name == "logreg_test"
        assert serving_scorer.threshold == pytest.approx(0.58)
        assert len(serving_scorer.feature_set) > 0
        assert len(serving_scorer.input_columns) > 0

    def test_threshold_override_wins_over_manifest(
        self, serving_artifacts: ServingArtifacts
    ) -> None:
        scorer = ChurnScorer(
            model_name=serving_artifacts.model_name,
            models_dir=serving_artifacts.models_dir,
            preprocessor_path=serving_artifacts.preprocessor_path,
            threshold=0.9,
        )
        assert scorer.threshold == pytest.approx(0.9)

    def test_missing_model_raises(self, serving_artifacts: ServingArtifacts) -> None:
        with pytest.raises(FileNotFoundError):
            ChurnScorer(
                model_name="no_existe",
                models_dir=serving_artifacts.models_dir,
                preprocessor_path=serving_artifacts.preprocessor_path,
            )

    def test_missing_preprocessor_raises(self, serving_artifacts: ServingArtifacts) -> None:
        with pytest.raises(FileNotFoundError, match="Preprocesador"):
            ChurnScorer(
                model_name=serving_artifacts.model_name,
                models_dir=serving_artifacts.models_dir,
                preprocessor_path=serving_artifacts.models_dir / "no_existe.joblib",
            )


class TestPrediction:
    def test_probabilities_are_valid(self, serving_scorer: ChurnScorer) -> None:
        predictions = serving_scorer.predict_payloads([make_payload()])
        assert len(predictions) == 1
        pred = predictions[0]
        assert 0.0 <= pred.probability <= 1.0
        assert pred.prediction in (0, 1)
        assert pred.customerID == "9237-HQITU"

    def test_label_consistent_with_threshold(self, serving_scorer: ChurnScorer) -> None:
        pred = serving_scorer.predict_payloads([make_payload()])[0]
        expected = "churn" if pred.probability >= serving_scorer.threshold else "no_churn"
        assert pred.label == expected
        assert pred.prediction == int(pred.probability >= serving_scorer.threshold)

    def test_batch_preserves_order(self, serving_scorer: ChurnScorer) -> None:
        payloads = [
            make_payload(customerID="AAAA-00001"),
            make_payload(customerID="BBBB-00002", tenure=60, Contract="Two year"),
            make_payload(customerID="CCCC-00003", tenure=12),
        ]
        predictions = serving_scorer.predict_payloads(payloads)
        assert [p.customerID for p in predictions] == ["AAAA-00001", "BBBB-00002", "CCCC-00003"]

    def test_new_customer_with_null_total_charges(self, serving_scorer: ChurnScorer) -> None:
        """tenure=0 con TotalCharges nulo (caso real del dataset) no debe fallar."""
        payload = make_payload(
            customerID="4472-LVYGI",
            tenure=0,
            PhoneService="No",
            MultipleLines="No phone service",
            InternetService="No",
            OnlineSecurity="No internet service",
            OnlineBackup="No internet service",
            DeviceProtection="No internet service",
            TechSupport="No internet service",
            StreamingTV="No internet service",
            StreamingMovies="No internet service",
            Contract="Two year",
            PaymentMethod="Bank transfer (automatic)",
            TotalCharges=None,
        )
        pred = serving_scorer.predict_payloads([payload])[0]
        assert 0.0 <= pred.probability <= 1.0

    def test_missing_column_raises_value_error(self, serving_scorer: ChurnScorer) -> None:
        df = serving_scorer._payloads_to_frame([make_payload()]).drop(columns=["Contract"])
        with pytest.raises(ValueError, match="Faltan columnas"):
            serving_scorer.predict_proba_frame(df)


class TestRiskBands:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.95, "high"),
            (0.58, "high"),  # límite inferior de high == threshold
            (0.5, "medium"),
            (0.29, "medium"),  # límite inferior de medium == threshold / 2
            (0.28, "low"),
            (0.01, "low"),
        ],
    )
    def test_band_boundaries(
        self, serving_scorer: ChurnScorer, probability: float, expected: str
    ) -> None:
        assert serving_scorer.risk_band(probability) == expected


class TestPayloadValidation:
    def test_invalid_contract_rejected(self) -> None:
        with pytest.raises(ValueError, match="Contract"):
            make_payload(Contract="Three year")

    def test_phone_lines_inconsistency_rejected(self) -> None:
        with pytest.raises(ValueError, match="MultipleLines"):
            make_payload(PhoneService="No", MultipleLines="Yes")

    def test_internet_addons_inconsistency_rejected(self) -> None:
        with pytest.raises(ValueError, match="add-ons"):
            make_payload(InternetService="No", OnlineSecurity="Yes")

    def test_tenure_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="tenure"):
            make_payload(tenure=100)
