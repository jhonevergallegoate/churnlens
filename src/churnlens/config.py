"""Configuración tipada del proyecto.

Toda la configuración se centraliza en `Settings`, que se hidrata desde:

1. Variables de entorno reales.
2. El archivo `.env` en la raíz del proyecto (si existe).
3. Los defaults definidos en esta clase.

Esto permite cambiar el comportamiento del pipeline sin tocar el código,
y facilita la integración con CI, contenedores y pipelines de despliegue.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# La raíz del proyecto se infiere desde la ubicación de este archivo:
#   src/churnlens/config.py  →  raíz = parents[2]
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuración global del proyecto ChurnLens."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Nivel de logging raíz.")
    log_format: str = Field(default="console", description="'console' o 'json'.")

    # ------------------------------------------------------------------
    # Datos
    # ------------------------------------------------------------------
    data_url: str = Field(
        default=(
            "https://raw.githubusercontent.com/IBM/"
            "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        ),
        alias="CHURNLENS_DATA_URL",
        description="URL pública del dataset Telco Customer Churn.",
    )
    raw_filename: str = Field(
        default="telco_customer_churn.csv",
        alias="CHURNLENS_RAW_FILENAME",
    )
    raw_md5_expected: str = Field(
        default="",
        alias="CHURNLENS_RAW_MD5",
        description="MD5 esperado. Si está vacío, se omite la verificación.",
    )

    # ------------------------------------------------------------------
    # Rutas
    # ------------------------------------------------------------------
    project_root: Path = Field(default=PROJECT_ROOT)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    # ------------------------------------------------------------------
    # Reproducibilidad
    # ------------------------------------------------------------------
    random_seed: int = Field(default=42, alias="CHURNLENS_RANDOM_SEED")

    # ------------------------------------------------------------------
    # Configuración de Pydantic
    # ------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Directorio donde viven los datos crudos inmutables."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Directorio donde viven los datos validados intermedios."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Directorio donde viven los datos listos para modelar."""
        return self.data_dir / "processed"

    @property
    def raw_csv_path(self) -> Path:
        """Ruta absoluta del CSV crudo."""
        return self.raw_dir / self.raw_filename

    @property
    def interim_parquet_path(self) -> Path:
        """Ruta absoluta del parquet validado."""
        return self.interim_dir / self.raw_filename.replace(".csv", ".parquet")

    @property
    def checksums_path(self) -> Path:
        """Ruta del archivo de auditoría de hashes."""
        return self.raw_dir / ".checksums.json"

    # ------------------------------------------------------------------
    # Validadores
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        """Asegura que el nivel de log sea válido y esté en mayúsculas."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            msg = f"log_level inválido: {v!r}. Debe ser uno de {sorted(valid)}."
            raise ValueError(msg)
        return v_upper

    @field_validator("log_format")
    @classmethod
    def _normalize_log_format(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in {"console", "json"}:
            msg = f"log_format inválido: {v!r}. Debe ser 'console' o 'json'."
            raise ValueError(msg)
        return v_lower

    def ensure_data_dirs(self) -> None:
        """Crea las carpetas estándar de datos si no existen."""
        for d in (self.raw_dir, self.interim_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


# Instancia única usada en toda la aplicación.
settings = Settings()
