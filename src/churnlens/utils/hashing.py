"""Utilidades de hashing para auditar la integridad de los datos crudos."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

UTC = timezone.utc

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class ChecksumRecord(TypedDict):
    """Entrada de auditoría para un archivo descargado."""

    md5: str
    sha256: str
    bytes: int
    downloaded_at: str


def compute_md5(path: Path) -> str:
    """Calcula el hash MD5 de un archivo.

    Args:
        path: Ruta del archivo.

    Returns:
        Hash hex de 32 caracteres.
    """
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo.

    Args:
        path: Ruta del archivo.

    Returns:
        Hash hex de 64 caracteres.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def build_checksum_record(path: Path) -> ChecksumRecord:
    """Construye un registro completo de checksums para un archivo dado."""
    return ChecksumRecord(
        md5=compute_md5(path),
        sha256=compute_sha256(path),
        bytes=path.stat().st_size,
        downloaded_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def save_checksums(records: dict[str, ChecksumRecord], path: Path) -> None:
    """Persiste un mapa de checksums a JSON.

    Args:
        records: Diccionario `nombre_archivo -> ChecksumRecord`.
        path:    Ruta destino del JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def load_checksums(path: Path) -> dict[str, ChecksumRecord]:
    """Lee un mapa de checksums desde disco.

    Si el archivo no existe, retorna un diccionario vacío.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def verify_md5(path: Path, expected_md5: str) -> bool:
    """Verifica el MD5 de un archivo contra un valor esperado.

    La comparación es case-insensitive y robusta a espacios.

    Args:
        path:         Archivo a verificar.
        expected_md5: Hash MD5 esperado (vacío = se omite la verificación).

    Returns:
        True si coincide o si `expected_md5` es vacío; False en caso contrario.
    """
    expected = expected_md5.strip().lower()
    if not expected:
        return True
    return compute_md5(path).lower() == expected
