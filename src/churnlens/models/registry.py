"""Registro local de modelos entrenados.

Cada modelo se persiste como un par:

* ``models/<nombre>.joblib`` — el estimador serializado con joblib.
* ``models/<nombre>.metadata.json`` — manifiesto auditable con:

  * ``name``, ``algorithm``, ``created_at`` (UTC ISO-8601).
  * ``train_path``, ``val_path``, ``feature_set``.
  * ``hash_train``, ``hash_val`` — SHA-256 de los parquet usados.
  * ``hash_model`` — SHA-256 del joblib generado (anti-tampering).
  * ``metrics`` — métricas reportadas (train / val).
  * ``hyperparameters`` — diccionario plano de hiperparámetros.

El registro no usa MLflow ni otra capa pesada: se mantiene auditado y
versionable por git para reproducibilidad académica.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from churnlens.config import settings as default_settings
from churnlens.utils.hashing import compute_sha256

DEFAULT_MODELS_DIRNAME: str = "models"


@dataclass(frozen=True)
class ModelEntry:
    """Entrada del registro: estimador + manifiesto."""

    name: str
    model_path: Path
    metadata_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def algorithm(self) -> str:
        """Algoritmo declarado en el manifiesto."""
        return str(self.metadata.get("algorithm", "?"))

    @property
    def created_at(self) -> str:
        """Marca temporal de creación (UTC ISO-8601)."""
        return str(self.metadata.get("created_at", "?"))

    @property
    def metrics(self) -> dict[str, Any]:
        """Métricas reportadas (puede estar vacío)."""
        return dict(self.metadata.get("metrics") or {})


def _models_dir(models_dir: Path | str | None) -> Path:
    if models_dir is not None:
        return Path(models_dir)
    return default_settings.models_dir


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def save_model(
    model: Any,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    models_dir: Path | str | None = None,
) -> ModelEntry:
    """Persiste un estimador y su manifiesto.

    Args:
        model: estimador entrenado (cualquier objeto picklable).
        name: nombre lógico del modelo (sin extensión).
        metadata: dict con campos a registrar; se enriquece con
            ``created_at`` y ``hash_model``.
        models_dir: directorio destino (default = ``<proyecto>/models``).

    Returns:
        :class:`ModelEntry` con las rutas y el manifiesto enriquecido.
    """
    if not name or "/" in name or name.startswith("."):
        msg = f"Nombre de modelo inválido: {name!r}"
        raise ValueError(msg)

    target_dir = _models_dir(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / f"{name}.joblib"
    metadata_path = target_dir / f"{name}.metadata.json"

    joblib.dump(model, model_path)

    enriched: dict[str, Any] = dict(metadata or {})
    enriched.setdefault("name", name)
    enriched.setdefault("algorithm", type(model).__name__)
    enriched["created_at"] = _now_iso()
    enriched["hash_model"] = compute_sha256(model_path)

    metadata_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return ModelEntry(
        name=name,
        model_path=model_path,
        metadata_path=metadata_path,
        metadata=enriched,
    )


def load_model(
    name: str,
    *,
    models_dir: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Carga un estimador y su manifiesto desde el registro.

    Args:
        name: nombre lógico (sin extensión).
        models_dir: directorio fuente.

    Returns:
        Tupla ``(model, metadata)``.

    Raises:
        FileNotFoundError: si el modelo o el manifiesto no existen.
    """
    target_dir = _models_dir(models_dir)
    model_path = target_dir / f"{name}.joblib"
    metadata_path = target_dir / f"{name}.metadata.json"
    if not model_path.exists():
        msg = f"Modelo '{name}' no encontrado en {target_dir}."
        raise FileNotFoundError(msg)
    if not metadata_path.exists():
        msg = f"Manifiesto '{name}.metadata.json' no encontrado en {target_dir}."
        raise FileNotFoundError(msg)

    model = joblib.load(model_path)
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def list_models(*, models_dir: Path | str | None = None) -> list[ModelEntry]:
    """Devuelve todas las entradas del registro.

    Returns:
        Lista de :class:`ModelEntry` ordenada por nombre.
    """
    target_dir = _models_dir(models_dir)
    if not target_dir.exists():
        return []
    entries: list[ModelEntry] = []
    for meta_path in sorted(target_dir.glob("*.metadata.json")):
        name = meta_path.name.removesuffix(".metadata.json")
        model_path = target_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        entries.append(
            ModelEntry(
                name=name,
                model_path=model_path,
                metadata_path=meta_path,
                metadata=metadata,
            )
        )
    return entries


def best_model_by(
    metric: str,
    *,
    higher_is_better: bool = True,
    split: str = "val",
    models_dir: Path | str | None = None,
) -> ModelEntry | None:
    """Devuelve la entrada con mejor valor de `metric` en `split`.

    Args:
        metric: nombre de la métrica (ej. ``"pr_auc"``).
        higher_is_better: si ``True`` (default), elige el máximo.
        split: ``"val"`` (default) o ``"train"`` — qué bloque de
            métricas inspeccionar.
        models_dir: directorio fuente.

    Returns:
        :class:`ModelEntry` ganadora, o ``None`` si no hay candidatos.
    """
    entries = list_models(models_dir=models_dir)
    candidates: list[tuple[float, ModelEntry]] = []
    for entry in entries:
        metrics_split = (entry.metrics.get(split) or {}) if isinstance(entry.metrics, dict) else {}
        value = metrics_split.get(metric) if isinstance(metrics_split, dict) else None
        if isinstance(value, (int, float)):
            candidates.append((float(value), entry))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[0], reverse=higher_is_better)
    return candidates[0][1]
