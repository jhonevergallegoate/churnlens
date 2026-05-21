"""Tests para `churnlens.features.preprocessing` y `churnlens.features.splits`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnlens.features.engineering import add_engineered_features
from churnlens.features.preprocessing import (
    BINARY_CATEGORICAL_COLS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    binarize_target,
    build_preprocessor,
)
from churnlens.features.splits import stratified_split


class TestBinarizeTarget:
    def test_maps_yes_to_one_no_to_zero(self) -> None:
        s = pd.Series(["Yes", "No", "No", "Yes"])
        out = binarize_target(s)
        assert list(out) == [1, 0, 0, 1]
        assert out.dtype == np.int8

    def test_rejects_unknown_values(self) -> None:
        with pytest.raises(ValueError, match="fuera de dominio"):
            binarize_target(pd.Series(["Yes", "Maybe"]))


class TestBuildPreprocessor:
    def test_transform_produces_dense_numeric_array(
        self, validated_synthetic: pd.DataFrame
    ) -> None:
        df = add_engineered_features(validated_synthetic).drop(columns=["customerID"])
        df_features = df.drop(columns=["Churn"])
        preprocessor = build_preprocessor(include_engineered=True)
        arr = preprocessor.fit_transform(df_features)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2
        assert arr.shape[0] == len(df_features)
        assert np.isfinite(arr).all()

    def test_feature_names_unique_and_match_columns(
        self, validated_synthetic: pd.DataFrame
    ) -> None:
        df = add_engineered_features(validated_synthetic).drop(columns=["customerID"])
        df_features = df.drop(columns=["Churn"])
        preprocessor = build_preprocessor()
        arr = preprocessor.fit_transform(df_features)
        names = list(preprocessor.get_feature_names_out())
        assert len(names) == arr.shape[1]
        assert len(set(names)) == len(names)
        for col in (*NUMERIC_COLS, *ORDINAL_COLS, *BINARY_CATEGORICAL_COLS):
            assert col in names

    def test_imputes_total_charges_missing(self) -> None:
        # Construye un mini-DataFrame donde TotalCharges tiene NaN.
        df = pd.DataFrame(
            {
                "tenure": [0, 12, 24, 36],
                "MonthlyCharges": [25.0, 50.0, 75.0, 100.0],
                "TotalCharges": [np.nan, 600.0, 1800.0, 3600.0],
                "Contract": ["Month-to-month"] * 4,
                "tenure_bucket": pd.Categorical(
                    ["0-12m", "0-12m", "13-24m", "25-48m"],
                    categories=["0-12m", "13-24m", "25-48m", "49-72m"],
                    ordered=True,
                ),
                "gender": ["Male"] * 4,
                "Partner": ["No"] * 4,
                "Dependents": ["No"] * 4,
                "PhoneService": ["Yes"] * 4,
                "PaperlessBilling": ["Yes"] * 4,
                "MultipleLines": ["No"] * 4,
                "InternetService": ["DSL"] * 4,
                "OnlineSecurity": ["No"] * 4,
                "OnlineBackup": ["No"] * 4,
                "DeviceProtection": ["No"] * 4,
                "TechSupport": ["No"] * 4,
                "StreamingTV": ["No"] * 4,
                "StreamingMovies": ["No"] * 4,
                "PaymentMethod": ["Mailed check"] * 4,
                "services_count": np.array([1, 1, 1, 1], dtype="int8"),
                "has_internet": [True] * 4,
                "has_phone": [True] * 4,
                "auto_payment": [False] * 4,
                "avg_monthly_spend": [25.0, 50.0, 75.0, 100.0],
                "monthly_spend_gap": [0.0, 0.0, 0.0, 0.0],
            }
        )
        preprocessor = build_preprocessor()
        arr = preprocessor.fit_transform(df)
        assert np.isfinite(arr).all()


class TestStratifiedSplit:
    def test_preserves_target_rate(self, validated_synthetic: pd.DataFrame) -> None:
        df = add_engineered_features(validated_synthetic).drop(columns=["customerID"])
        split = stratified_split(df, random_state=7)
        rates = split.target_rates
        ref = rates["train"]
        assert abs(rates["val"] - ref) < 0.05
        assert abs(rates["test"] - ref) < 0.05

    def test_shapes_sum_to_original(self, validated_synthetic: pd.DataFrame) -> None:
        df = add_engineered_features(validated_synthetic).drop(columns=["customerID"])
        split = stratified_split(df)
        total = split.X_train.shape[0] + split.X_val.shape[0] + split.X_test.shape[0]
        assert total == len(df)

    def test_no_overlap_between_splits(self, validated_synthetic: pd.DataFrame) -> None:
        df = add_engineered_features(validated_synthetic).drop(columns=["customerID"])
        df = df.reset_index(drop=True).assign(_uid=lambda d: d.index.astype(int))
        split = stratified_split(df)
        train_ids = set(split.X_train["_uid"])
        val_ids = set(split.X_val["_uid"])
        test_ids = set(split.X_test["_uid"])
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_rejects_invalid_fractions(self, validated_synthetic: pd.DataFrame) -> None:
        df = add_engineered_features(validated_synthetic).drop(columns=["customerID"])
        with pytest.raises(ValueError, match="fracciones"):
            stratified_split(df, test_size=0.6, val_size=0.5)
