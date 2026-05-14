"""Tests para `churnlens.data.schema`."""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from churnlens.data.loader import TelcoChurnLoader
from churnlens.data.schema import RAW_SCHEMA


def test_schema_validates_sample(sample_dataframe: pd.DataFrame) -> None:
    """El esquema debe validar correctamente el CSV de muestra ya casteado."""
    df = TelcoChurnLoader._coerce_types(sample_dataframe)
    validated = RAW_SCHEMA.validate(df, lazy=True)
    assert len(validated) == len(df)


def test_schema_detects_bad_gender(sample_dataframe: pd.DataFrame) -> None:
    df = TelcoChurnLoader._coerce_types(sample_dataframe)
    df["gender"] = df["gender"].astype("object")
    df.loc[df.index[0], "gender"] = "Unknown"
    df["gender"] = df["gender"].astype("category")
    with pytest.raises(pa_errors.SchemaErrors):
        RAW_SCHEMA.validate(df, lazy=True)


def test_schema_detects_bad_tenure_range(sample_dataframe: pd.DataFrame) -> None:
    df = TelcoChurnLoader._coerce_types(sample_dataframe)
    df.loc[df.index[0], "tenure"] = 99
    with pytest.raises(pa_errors.SchemaErrors):
        RAW_SCHEMA.validate(df, lazy=True)


def test_schema_detects_phone_lines_inconsistency(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Si PhoneService=No, MultipleLines debe ser 'No phone service'."""
    df = TelcoChurnLoader._coerce_types(sample_dataframe)
    idx = df.index[0]
    df["PhoneService"] = df["PhoneService"].astype("object")
    df["MultipleLines"] = df["MultipleLines"].astype("object")
    df.loc[idx, "PhoneService"] = "No"
    df.loc[idx, "MultipleLines"] = "Yes"
    df["PhoneService"] = df["PhoneService"].astype("category")
    df["MultipleLines"] = df["MultipleLines"].astype("category")
    with pytest.raises(pa_errors.SchemaErrors):
        RAW_SCHEMA.validate(df, lazy=True)


def test_schema_detects_duplicate_customer_id(
    sample_dataframe: pd.DataFrame,
) -> None:
    df = TelcoChurnLoader._coerce_types(sample_dataframe)
    df.loc[df.index[1], "customerID"] = df.loc[df.index[0], "customerID"]
    with pytest.raises(pa_errors.SchemaErrors):
        RAW_SCHEMA.validate(df, lazy=True)
