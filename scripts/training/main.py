"""Punto de entrada oficial de la Fase 3 (TDSP · entrenamiento).

Implementa el entregable **"Código del modelamiento"** exigido por la
rúbrica del Módulo 6 del Diplomado MLDS. Es un _wrapper_ delgado sobre
:func:`churnlens.features.selection.run_feature_selection` y
:func:`churnlens.models.train.train_models`, de modo que el script y la
CLI (`churnlens features select` + `churnlens model train`) producen
artefactos byte-equivalentes.

Uso:

```bash
python scripts/training/main.py
make train
```

Flujo:

1. Selección de features (cuatro técnicas + consenso top-k).
2. Entrenamiento de los 8 modelos del catálogo
   (:data:`churnlens.models.train.MODEL_SPECS`) con CV estratificada
   5-fold, sobre el set completo de features.
3. Persistencia de cada modelo + manifiesto en ``models/``.
4. Tablas comparativas y figuras PR/ROC en ``reports/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from churnlens.config import Settings
from churnlens.features.selection import (
    persist_feature_selection,
    run_feature_selection,
)
from churnlens.logger import get_logger
from churnlens.models.train import train_models

log = get_logger("scripts.training")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="training",
        description="Selección de features + entrenamiento de los modelos de Fase 3.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Tamaño del consenso top-k para la selección de features (default: 20).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Folds para validación cruzada estratificada (default: 5).",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Omite la fase de selección de features (asume artefactos previos).",
    )
    parser.add_argument(
        "--use-consensus",
        action="store_true",
        help="Restringe el entrenamiento al top-k del consenso de features.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Modelos a entrenar (default: todos). Ej: --models logreg_balanced lightgbm",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity a WARNING.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script de entrenamiento."""
    args = _parse_args(argv)
    settings = Settings(log_level="WARNING") if args.quiet else Settings()

    log.info("phase3_training_start")

    feature_subset: list[str] | None = None
    if not args.skip_selection:
        log.info("phase3_feature_selection_start", top_k=args.top_k)
        selection = run_feature_selection(k=args.top_k, random_state=settings.random_seed)
        persist_feature_selection(selection)
        log.info("phase3_feature_selection_done", top_k=selection.top_k)
        if args.use_consensus:
            feature_subset = selection.top_k

    artifacts = train_models(
        models=args.models,
        feature_subset=feature_subset,
        cv=args.cv,
        random_state=settings.random_seed,
    )
    log.info(
        "phase3_training_done",
        best_model=artifacts.best_model_name,
        n_models=len(artifacts.model_entries),
        paths=[str(p) for p in artifacts.paths.values()],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
