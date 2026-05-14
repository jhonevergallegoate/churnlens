"""Punto de entrada oficial de la Fase 1 (TDSP · data acquisition).

Este script implementa el entregable **"Código de carga de datos"** exigido
por la rúbrica del Módulo 6 del Diplomado MLDS (Universidad Nacional de
Colombia). Es un _wrapper_ delgado sobre la implementación canónica del
paquete `churnlens`, que reside en `src/churnlens/data/loader.py`.

Uso:

```bash
# Forma directa
python scripts/data_acquisition/main.py

# Equivalentes
make data
churnlens data download && churnlens data validate && churnlens data summary
```

Argumentos opcionales:

```
--url TEXT      URL fuente del CSV (sobrescribe el .env).
--output PATH   Ruta destino del CSV crudo.
--force         Re-descarga aunque el archivo exista.
--no-validate   Omite la validación con Pandera.
--quiet         Reduce verbosity.
```
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Asegura que el paquete sea importable cuando se ejecuta el script
# directamente desde la raíz del repo (`python scripts/data_acquisition/main.py`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse  # noqa: E402

from churnlens.config import Settings  # noqa: E402
from churnlens.data.loader import TelcoChurnLoader  # noqa: E402
from churnlens.logger import get_logger  # noqa: E402

log = get_logger("scripts.data_acquisition")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="data_acquisition",
        description="Descarga, valida y materializa el dataset Telco Customer Churn.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL fuente del CSV (sobrescribe la configuración).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta destino del archivo CSV crudo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vuelve a descargar incluso si el archivo ya existe.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="No ejecuta la validación contra el esquema Pandera.",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="No genera el parquet en data/interim/.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity a WARNING.",
    )
    return parser.parse_args(argv)


def _build_settings(args: argparse.Namespace) -> Settings:
    """Construye la configuración respetando los overrides de la CLI."""
    overrides: dict[str, str] = {}
    if args.quiet:
        overrides["log_level"] = "WARNING"
    if args.url is not None:
        overrides["data_url"] = args.url
    if args.output is not None:
        # `Settings.raw_filename` controla el archivo dentro de data/raw/,
        # mientras que `data_dir` controla el directorio base.
        output = args.output
        overrides["raw_filename"] = output.name
        overrides["data_dir"] = str(output.parent.parent) if output.parent.parent != Path() else str(output.parent)

    return Settings(**overrides) if overrides else Settings()


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = _parse_args(argv)
    settings = _build_settings(args)

    log.info("phase1_start", action="data_acquisition")
    loader = TelcoChurnLoader(settings=settings)

    # 1. Descarga
    csv_path = loader.download(force=args.force)
    log.info("download_done", path=str(csv_path))

    # 2. Validación
    if not args.no_validate:
        df = loader.load_validated()
        log.info("validation_done", n_rows=len(df), n_cols=df.shape[1])

    # 3. Materialización a parquet (recomendado por reproducibilidad).
    if not args.no_materialize:
        parquet_path = loader.materialize_interim()
        log.info("materialize_done", path=str(parquet_path))

    # 4. Resumen final
    summary = loader.summary()
    log.info("phase1_done", **{k: v for k, v in summary.to_dict().items() if k != "sha256"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
