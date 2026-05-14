"""Descarga, carga y validación del dataset Telco Customer Churn.

Este módulo expone una clase `TelcoChurnLoader` que orquesta el ciclo
completo de la Fase 1 de la metodología TDSP:

```
descarga ──▶ hash MD5/SHA-256 ──▶ casteo de tipos ──▶ Pandera ──▶ parquet
```

Diseñado para ser invocado tanto desde:

* La CLI (`churnlens data ...`)
* El script TDSP (`scripts/data_acquisition/main.py`)
* Notebooks (`notebooks/01_data_acquisition_eda.ipynb`)
* Otros módulos del paquete (importación directa)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests

from churnlens.config import Settings, settings as default_settings
from churnlens.data.schema import RAW_SCHEMA
from churnlens.logger import get_logger
from churnlens.utils.hashing import (
    build_checksum_record,
    load_checksums,
    save_checksums,
    verify_md5,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

log = get_logger(__name__)

_REQUEST_TIMEOUT_S: float = 60.0
_REQUEST_CHUNK_BYTES: int = 256 * 1024  # 256 KiB

# Tipos finales esperados después del casteo.
_COLUMN_DTYPES: Mapping[str, str] = {
    "customerID": "string",
    "gender": "category",
    "SeniorCitizen": "int8",
    "Partner": "category",
    "Dependents": "category",
    "tenure": "int16",
    "PhoneService": "category",
    "MultipleLines": "category",
    "InternetService": "category",
    "OnlineSecurity": "category",
    "OnlineBackup": "category",
    "DeviceProtection": "category",
    "TechSupport": "category",
    "StreamingTV": "category",
    "StreamingMovies": "category",
    "PaperlessBilling": "category",
    "PaymentMethod": "category",
    "MonthlyCharges": "float32",
    "TotalCharges": "float32",
    "Churn": "category",
}

_CONTRACT_ORDER: list[str] = ["Month-to-month", "One year", "Two year"]


@dataclass(frozen=True)
class DataSummary:
    """Resumen ejecutivo del dataset cargado."""

    n_rows: int
    n_cols: int
    target_pos_rate: float
    n_missing_total_charges: int
    md5: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Convierte el resumen a un diccionario serializable."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target_pos_rate": round(self.target_pos_rate, 4),
            "n_missing_total_charges": self.n_missing_total_charges,
            "md5": self.md5,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


class TelcoChurnLoader:
    """Orquestador de carga del dataset Telco Customer Churn.

    Args:
        settings: Configuración del proyecto. Si se omite, se usa la instancia
                  global de `churnlens.config.settings`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.settings.ensure_data_dirs()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def download(self, *, force: bool = False) -> Path:
        """Descarga el CSV crudo a `data/raw/` si no existe (o si `force=True`).

        Args:
            force: si es True, se vuelve a descargar aunque el archivo ya exista.

        Returns:
            Ruta al archivo CSV en disco.
        """
        out_path = self.settings.raw_csv_path
        if out_path.exists() and not force:
            log.info(
                "raw_already_present",
                path=str(out_path),
                bytes=out_path.stat().st_size,
            )
            self._record_checksums(out_path)
            return out_path

        url = self.settings.data_url
        log.info("downloading_raw", url=url, destination=str(out_path))

        with requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT_S) as response:
            response.raise_for_status()
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=_REQUEST_CHUNK_BYTES):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(out_path)

        log.info(
            "download_complete",
            path=str(out_path),
            bytes=out_path.stat().st_size,
        )

        # Verifica MD5 esperado (si está configurado).
        if not verify_md5(out_path, self.settings.raw_md5_expected):
            msg = (
                f"MD5 del archivo descargado no coincide con el esperado "
                f"({self.settings.raw_md5_expected!r})."
            )
            raise RuntimeError(msg)

        self._record_checksums(out_path)
        return out_path

    def load_raw(self) -> pd.DataFrame:
        """Lee el CSV crudo y devuelve un DataFrame **sin** validar ni castear."""
        path = self.settings.raw_csv_path
        if not path.exists():
            msg = (
                f"No se encontró el archivo crudo en {path}. "
                "Ejecuta `churnlens data download` primero."
            )
            raise FileNotFoundError(msg)

        log.debug("reading_raw_csv", path=str(path))
        return pd.read_csv(path, dtype="object")

    def load_validated(self) -> pd.DataFrame:
        """Carga, castea tipos y valida el dataset contra el esquema Pandera.

        Returns:
            DataFrame con tipos estables y validado.
        """
        df = self.load_raw()
        df = self._coerce_types(df)
        log.info("validating_schema", n_rows=len(df), n_cols=df.shape[1])
        validated = RAW_SCHEMA.validate(df, lazy=True)
        log.info("schema_ok", n_rows=len(validated))
        return validated

    def materialize_interim(self) -> Path:
        """Persiste el dataset validado como parquet en `data/interim/`.

        Returns:
            Ruta al archivo parquet generado.
        """
        df = self.load_validated()
        out_path = self.settings.interim_parquet_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        log.info(
            "interim_written",
            path=str(out_path),
            n_rows=len(df),
            bytes=out_path.stat().st_size,
        )
        return out_path

    def summary(self) -> DataSummary:
        """Calcula el resumen ejecutivo del dataset crudo descargado."""
        df = self.load_validated()
        checksums = load_checksums(self.settings.checksums_path)
        record: dict[str, object] = dict(checksums.get(self.settings.raw_filename, {}))
        return DataSummary(
            n_rows=int(len(df)),
            n_cols=int(df.shape[1]),
            target_pos_rate=float((df["Churn"] == "Yes").mean()),
            n_missing_total_charges=int(df["TotalCharges"].isna().sum()),
            md5=str(record.get("md5", "")),
            sha256=str(record.get("sha256", "")),
            bytes=int(record.get("bytes", 0)),  # type: ignore[call-overload]
        )

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica el casteo de tipos canónico del proyecto.

        - Reemplaza `" "` por `NaN` en `TotalCharges`.
        - Convierte `TotalCharges` a `float32`.
        - Castea categóricas y enteros pequeños.
        - Marca `Contract` como categórica ordenada.
        """
        df = df.copy()

        # TotalCharges trae 11 cadenas vacías que representan NaN.
        df["TotalCharges"] = (
            df["TotalCharges"].astype("string").str.strip().replace({"": pd.NA})
        )
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="raise").astype(
            "float32"
        )

        # SeniorCitizen + tenure llegan como string desde dtype="object".
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="raise").astype(
            "int8"
        )
        df["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype("int16")
        df["MonthlyCharges"] = pd.to_numeric(
            df["MonthlyCharges"], errors="raise"
        ).astype("float32")

        # Castea categóricas.
        for col, dtype in _COLUMN_DTYPES.items():
            if dtype == "category" and col in df.columns:
                df[col] = df[col].astype("category")

        # Contract como categoría ordenada.
        df["Contract"] = pd.Categorical(
            df["Contract"], categories=_CONTRACT_ORDER, ordered=True
        )

        df["customerID"] = df["customerID"].astype("string")
        return df

    def _record_checksums(self, path: Path) -> None:
        """Calcula y persiste hashes MD5/SHA-256 del archivo descargado."""
        records = load_checksums(self.settings.checksums_path)
        records[path.name] = build_checksum_record(path)
        save_checksums(records, self.settings.checksums_path)
        log.info(
            "checksums_saved",
            md5=records[path.name]["md5"],
            sha256=records[path.name]["sha256"],
            path=str(self.settings.checksums_path),
        )
