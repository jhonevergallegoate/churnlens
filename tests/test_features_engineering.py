"""Tests para `churnlens.features.engineering`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnlens.features.engineering import (
    ADDON_COLS,
    SERVICE_COLS,
    TENURE_BUCKET_LABELS,
    add_engineered_features,
)


class TestAddEngineeredFeatures:
    def test_does_not_mutate_input(self, validated_sample: pd.DataFrame) -> None:
        before = validated_sample.copy()
        add_engineered_features(validated_sample)
        pd.testing.assert_frame_equal(validated_sample, before)

    def test_adds_expected_columns(self, validated_sample: pd.DataFrame) -> None:
        out = add_engineered_features(validated_sample)
        expected = {
            "tenure_bucket",
            "services_count",
            "has_internet",
            "has_phone",
            "auto_payment",
            "avg_monthly_spend",
            "monthly_spend_gap",
        }
        assert expected.issubset(out.columns)

    def test_tenure_bucket_is_ordered_categorical(self, validated_sample: pd.DataFrame) -> None:
        out = add_engineered_features(validated_sample)
        assert isinstance(out["tenure_bucket"].dtype, pd.CategoricalDtype)
        assert out["tenure_bucket"].cat.ordered
        assert list(out["tenure_bucket"].cat.categories) == list(TENURE_BUCKET_LABELS)

    def test_tenure_bucket_values_known_rows(self, validated_sample: pd.DataFrame) -> None:
        out = add_engineered_features(validated_sample)
        # Filas de la muestra: tenure = [1, 34, 2, 45, 2, 0]
        assert list(out["tenure_bucket"].astype(str)) == [
            "0-12m",
            "25-48m",
            "0-12m",
            "25-48m",
            "0-12m",
            "0-12m",
        ]

    def test_services_count_bounds(self, validated_synthetic: pd.DataFrame) -> None:
        out = add_engineered_features(validated_synthetic)
        # Hay 8 columnas que pueden ser "Yes": PhoneService, MultipleLines + 6 addons.
        assert out["services_count"].between(0, len(SERVICE_COLS)).all()
        assert out["services_count"].dtype == np.int8

    def test_has_internet_consistent_with_internet_service(
        self, validated_synthetic: pd.DataFrame
    ) -> None:
        out = add_engineered_features(validated_synthetic)
        no_internet_mask = out["InternetService"].astype(str) == "No"
        assert not out.loc[no_internet_mask, "has_internet"].any()
        assert out.loc[~no_internet_mask, "has_internet"].all()

    def test_auto_payment_marks_only_automatic_methods(
        self, validated_synthetic: pd.DataFrame
    ) -> None:
        out = add_engineered_features(validated_synthetic)
        auto_methods = {"Bank transfer (automatic)", "Credit card (automatic)"}
        is_auto = out["PaymentMethod"].astype(str).isin(auto_methods)
        pd.testing.assert_series_equal(
            out["auto_payment"].rename(None).reset_index(drop=True),
            is_auto.rename(None).reset_index(drop=True),
            check_names=False,
        )

    def test_avg_monthly_spend_is_finite_even_with_zero_tenure(
        self, validated_sample: pd.DataFrame
    ) -> None:
        out = add_engineered_features(validated_sample)
        assert np.isfinite(out["avg_monthly_spend"]).all()

    def test_addons_aggregate_into_services_count(self, validated_synthetic: pd.DataFrame) -> None:
        out = add_engineered_features(validated_synthetic)
        manual = (
            (out["PhoneService"].astype(str) == "Yes").astype("int8")
            + (out["MultipleLines"].astype(str) == "Yes").astype("int8")
            + sum((out[c].astype(str) == "Yes").astype("int8") for c in ADDON_COLS)
        )
        assert (out["services_count"].astype("int8") == manual).all()

    def test_raises_when_required_column_missing(self, validated_sample: pd.DataFrame) -> None:
        broken = validated_sample.drop(columns=["PaymentMethod"])
        with pytest.raises(KeyError, match="PaymentMethod"):
            add_engineered_features(broken)
