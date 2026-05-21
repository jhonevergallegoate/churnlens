"""Punto de entrada oficial de la Fase 2 (TDSP · preprocesamiento).

Implementa el entregable **"Código de preprocesamiento"** exigido por la
rúbrica del Módulo 6 del Diplomado MLDS. Es un _wrapper_ delgado sobre
`churnlens.features.pipeline.run_preprocessing`, lo que garantiza que el
script y la CLI producen artefactos byte-equivalentes.

Uso:

```bash
python scripts/preprocessing/main.py
make preprocess
churnlens preprocess run
```
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
from churnlens.features.pipeline import run_preprocessing
from churnlens.logger import get_logger

log = get_logger("scripts.preprocessing")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preprocessing",
        description="Preprocesa el dataset Telco Churn y materializa train/val/test.",
    )
    parser.add_argument(
        "--no-engineered",
        action="store_true",
        help="Omite las features derivadas (tenure_bucket, services_count, etc.).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity a WARNING.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = _parse_args(argv)
    settings = Settings(log_level="WARNING") if args.quiet else Settings()

    log.info("phase2_preprocess_start")
    artifacts = run_preprocessing(
        settings=settings,
        include_engineered=not args.no_engineered,
    )
    log.info("phase2_preprocess_done", **artifacts.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
