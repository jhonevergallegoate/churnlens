"""Punto de entrada oficial de la Fase 5 (TDSP · evaluación final).

Implementa la **auditoría de equidad algorítmica** comprometida en
``docs/governance/ethics_and_fairness.md`` (§3): métricas por subgrupo
(selection rate, TPR, FPR, precision, ECE) e indicadores agregados
(Disparate Impact, Demographic Parity diff, Equalized Odds diff) para
los atributos sensibles ``gender``, ``SeniorCitizen``, ``Partner`` y
``Dependents``, evaluados sobre el held-out ``test.parquet`` con el
threshold operativo del modelo.

Genera:

* ``reports/tables/fairness_groups_<model>.csv``  — métricas por subgrupo.
* ``reports/tables/fairness_summary_<model>.json`` — indicadores + veredictos.
* ``reports/figures/fairness_audit_<model>.png``   — comparativa visual.

La auditoría es **informativa por defecto** (exit code 0 aunque haya
umbrales violados); con ``--strict`` devuelve 2 si algún atributo queda
fuera de los umbrales declarados — útil como gate de CI.

Uso:

```bash
python scripts/evaluation/fairness_audit.py
make fairness
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from churnlens.config import Settings
from churnlens.logger import get_logger
from churnlens.models.fairness import run_fairness_audit

log = get_logger("scripts.evaluation.fairness")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fairness_audit",
        description="Auditoría de fairness del modelo desplegado (Fase 5).",
    )
    parser.add_argument(
        "--model",
        default="logreg_l1",
        help="Nombre del modelo registrado a auditar. Default: logreg_l1.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold operativo; default = el sintonizado del manifest.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Devuelve exit code 2 si algún atributo viola los umbrales.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity a WARNING.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script de auditoría de fairness."""
    args = _parse_args(argv)
    settings = Settings(log_level="WARNING") if args.quiet else Settings()

    log.info("phase5_fairness_start", model=args.model)
    try:
        result = run_fairness_audit(
            model_name=args.model,
            threshold=args.threshold,
            settings=settings,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("phase5_fairness_failed", error=str(exc))
        return 1

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    log.info(
        "phase5_fairness_done",
        groups=str(result.groups_path),
        summary=str(result.summary_path),
        figure=str(result.figure_path),
        all_within_thresholds=result.all_within_thresholds,
    )
    if args.strict and not result.all_within_thresholds:
        log.warning(
            "phase5_fairness_thresholds_violated",
            attributes=[a for a, s in result.summary.items() if not s["within_thresholds"]],
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
