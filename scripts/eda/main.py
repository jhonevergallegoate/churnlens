"""Punto de entrada oficial de la Fase 2 (TDSP · análisis exploratorio).

Implementa el entregable **"Código de análisis exploratorio"** exigido
por la rúbrica del Módulo 6 del Diplomado MLDS. Es un _wrapper_ delgado
sobre `churnlens.eda.report.generate_eda_report`.

Genera 9 figuras PNG y 4 tablas CSV bajo `reports/figures/` y
`reports/tables/`.

Uso:

```bash
python scripts/eda/main.py
make eda
churnlens eda report
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
from churnlens.eda.report import generate_eda_report
from churnlens.logger import get_logger

log = get_logger("scripts.eda")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eda",
        description="Genera figuras y tablas del análisis exploratorio.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directorio de salida para las figuras PNG.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Directorio de salida para las tablas CSV.",
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

    log.info("phase2_eda_start")
    report = generate_eda_report(
        settings=settings,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )
    log.info(
        "phase2_eda_done",
        n_figures=len(report.figures),
        n_tables=len(report.tables),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
