"""Tests para `churnlens.eda.summary` y `churnlens.eda.report`."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # backend headless para CI

from churnlens.eda.report import generate_eda_report
from churnlens.eda.summary import (
    categorical_summary,
    churn_rate_by_category,
    cramers_v_vs_target,
    numeric_correlation,
    numeric_summary,
    target_distribution,
)
from churnlens.features.engineering import add_engineered_features


class TestNumericSummary:
    def test_basic_columns_present(self, validated_sample: pd.DataFrame) -> None:
        out = numeric_summary(validated_sample, columns=["tenure", "MonthlyCharges"])
        assert set(out.columns) >= {"count", "missing", "mean", "std", "min", "max"}
        assert set(out.index) == {"tenure", "MonthlyCharges"}

    def test_missing_count_matches(self, validated_sample: pd.DataFrame) -> None:
        out = numeric_summary(validated_sample, columns=["TotalCharges"])
        assert out.loc["TotalCharges", "missing"] == int(
            validated_sample["TotalCharges"].isna().sum()
        )


class TestCategoricalSummary:
    def test_top_freq_within_bounds(self, validated_sample: pd.DataFrame) -> None:
        out = categorical_summary(validated_sample, columns=["gender", "Contract"])
        assert (out["top_freq"] >= 0).all()
        assert (out["top_freq"] <= 1).all()


class TestTargetDistribution:
    def test_returns_proportions(self, validated_synthetic: pd.DataFrame) -> None:
        out = target_distribution(validated_synthetic)
        assert {"count", "pct"} <= set(out.columns)
        assert abs(out["pct"].sum() - 1.0) < 1e-6


class TestChurnRateByCategory:
    def test_rate_in_unit_interval(self, validated_synthetic: pd.DataFrame) -> None:
        out = churn_rate_by_category(validated_synthetic, "Contract")
        assert (out["churn_rate"] >= 0).all()
        assert (out["churn_rate"] <= 1).all()

    def test_is_reliable_flag(self, validated_synthetic: pd.DataFrame) -> None:
        out = churn_rate_by_category(validated_synthetic, "Contract", min_count=5)
        assert out["is_reliable"].dtype == bool


class TestCorrelations:
    def test_correlation_matrix_is_symmetric(self, validated_synthetic: pd.DataFrame) -> None:
        corr = numeric_correlation(
            validated_synthetic,
            columns=["tenure", "MonthlyCharges", "TotalCharges"],
        )
        assert corr.shape == (3, 3)
        # Simetría (con tolerancia para floats).
        for col in corr.columns:
            for row in corr.index:
                assert abs(corr.loc[row, col] - corr.loc[col, row]) < 1e-9

    def test_cramers_v_in_unit_interval(self, validated_synthetic: pd.DataFrame) -> None:
        v = cramers_v_vs_target(validated_synthetic, ["Contract", "gender"])
        assert v.between(0.0, 1.0).all()


class TestGenerateEdaReport:
    def test_produces_all_figures_and_tables(
        self,
        validated_synthetic: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        df = add_engineered_features(validated_synthetic)
        report = generate_eda_report(
            df=df,
            figures_dir=tmp_path / "figures",
            tables_dir=tmp_path / "tables",
        )
        assert len(report.figures) == 9
        assert len(report.tables) == 4
        for path in (*report.figures.values(), *report.tables.values()):
            assert path.exists()
            assert path.stat().st_size > 0
