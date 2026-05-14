"""Subpaquete de manejo de datos: descarga, validación y persistencia."""

from churnlens.data.loader import TelcoChurnLoader
from churnlens.data.schema import RAW_SCHEMA, build_raw_schema

__all__ = ["RAW_SCHEMA", "TelcoChurnLoader", "build_raw_schema"]
