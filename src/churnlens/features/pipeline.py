"""Orquestador end-to-end del preprocesamiento.

`run_preprocessing` toma el dataset validado (Fase 1) y produce:

* ``data/processed/train.parquet``
* ``data/processed/val.parquet``
* ``data/processed/test.parquet``
* ``data/processed/preprocessor.joblib``  (transformer ajustado a `train`)
* ``data/processed/feature_names.json``    (nombres post-transformación)
* ``data/processed/metadata.json``         (shapes, tasa de positivos, semilla)

El _transformer_ se ajusta **únicamente** sobre el conjunto de entrenamiento
para evitar _leakage_; luego se aplica a `val` y `test`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlens.config import Settings
from churnlens.config import settings as default_settings
from churnlens.data.loader import TelcoChurnLoader
from churnlens.features.engineering import add_engineered_features
from churnlens.features.preprocessing import (
    TARGET_COL,
    build_preprocessor,
)
from churnlens.features.splits import SplitResult, stratified_split
from churnlens.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Rutas de los artefactos producidos por `run_preprocessing`."""

    train_path: Path
    val_path: Path
    test_path: Path
    preprocessor_path: Path
    feature_names_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, str]:
        """Convierte a diccionario serializable."""
        return {
            "train": str(self.train_path),
            "val": str(self.val_path),
            "test": str(self.test_path),
            "preprocessor": str(self.preprocessor_path),
            "feature_names": str(self.feature_names_path),
            "metadata": str(self.metadata_path),
        }


def run_preprocessing(
    *,
    settings: Settings | None = None,
    include_engineered: bool = True,
) -> PreprocessingArtifacts:
    """Ejecuta el pipeline completo de preprocesamiento.

    Args:
        settings: configuración del proyecto.
        include_engineered: si ``True``, añade _features_ derivadas y las
            incluye en el `ColumnTransformer`.

    Returns:
        :class:`PreprocessingArtifacts` con las rutas de los archivos
        generados.
    """
    settings = settings or default_settings
    settings.ensure_data_dirs()
    out_dir = settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = TelcoChurnLoader(settings=settings)
    df = loader.load_validated()
    log.info("preprocess_loaded", n_rows=len(df), n_cols=df.shape[1])

    if include_engineered:
        df = add_engineered_features(df)
        log.info("preprocess_engineered", n_cols=df.shape[1])

    # `customerID` no entra al modelo, pero lo conservamos en los parquet
    # finales como columna auxiliar (útil para depuración).
    df = df.drop(columns=["customerID"])

    split = stratified_split(df, random_state=settings.random_seed)
    log.info("preprocess_split", shapes=split.shapes, rates=split.target_rates)

    preprocessor = build_preprocessor(include_engineered=include_engineered)
    x_train_arr = preprocessor.fit_transform(split.X_train)
    x_val_arr = preprocessor.transform(split.X_val)
    x_test_arr = preprocessor.transform(split.X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    train_df = _to_dataframe(x_train_arr, feature_names, split.y_train)
    val_df = _to_dataframe(x_val_arr, feature_names, split.y_val)
    test_df = _to_dataframe(x_test_arr, feature_names, split.y_test)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"
    train_df.to_parquet(train_path, engine="pyarrow", compression="snappy", index=False)
    val_df.to_parquet(val_path, engine="pyarrow", compression="snappy", index=False)
    test_df.to_parquet(test_path, engine="pyarrow", compression="snappy", index=False)

    preprocessor_path = out_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    feature_names_path = out_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    metadata = _build_metadata(split, feature_names, include_engineered, settings)
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info(
        "preprocess_materialized",
        train=str(train_path),
        val=str(val_path),
        test=str(test_path),
        n_features=len(feature_names),
    )

    return PreprocessingArtifacts(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        preprocessor_path=preprocessor_path,
        feature_names_path=feature_names_path,
        metadata_path=metadata_path,
    )


def _to_dataframe(arr: np.ndarray, feature_names: list[str], y: pd.Series) -> pd.DataFrame:
    """Empaqueta una matriz transformada + target en un DataFrame parquet-ready."""
    df_out = pd.DataFrame(arr, columns=feature_names).astype("float32")
    df_out[TARGET_COL] = y.to_numpy().astype("int8")
    return df_out


def _build_metadata(
    split: SplitResult,
    feature_names: list[str],
    include_engineered: bool,
    settings: Settings,
) -> dict[str, object]:
    """Construye el manifest JSON del split y del preprocesamiento."""
    y_all = pd.concat([split.y_train, split.y_val, split.y_test]).astype("int8")
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "random_seed": settings.random_seed,
        "include_engineered": include_engineered,
        "n_features_out": len(feature_names),
        "target_col": TARGET_COL,
        "target_positive_rate_global": float((y_all == 1).mean()),
        "splits": {
            name: {
                "n_rows": shape[0],
                "n_cols_in": shape[1],
                "positive_rate": rate,
            }
            for (name, shape), rate in zip(
                split.shapes.items(), split.target_rates.values(), strict=False
            )
        },
        "feature_names": feature_names,
    }
