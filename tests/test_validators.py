"""Tests para `churnlens.data.validators`."""

from __future__ import annotations

import pandas as pd

from churnlens.data.loader import TelcoChurnLoader
from churnlens.data.validators import compute_quality_report


def test_quality_report_basic(sample_dataframe: pd.DataFrame) -> None:
    df = TelcoChurnLoader._coerce_types(sample_dataframe)
    report = compute_quality_report(df)
    assert report.n_rows == len(df)
    assert report.n_cols == df.shape[1]
    assert report.n_duplicates == 0
    assert 0.0 <= report.target_positive_rate <= 1.0
    assert report.imbalance_ratio > 0


def test_quality_report_detects_missing_total_charges(
    sample_dataframe: pd.DataFrame,
) -> None:
    df = TelcoChurnLoader._coerce_types(sample_dataframe)
    report = compute_quality_report(df)
    assert report.missing_per_column.get("TotalCharges", 0) >= 1
