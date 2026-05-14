"""Logger estructurado para ChurnLens.

Wrapper delgado sobre `structlog` que respeta `Settings.log_level` y
`Settings.log_format`. En modo `console` produce salida legible para
humanos con colores; en modo `json` produce líneas serializadas listas
para herramientas de observabilidad (Loki, Datadog, Sentry, etc.).
"""

from __future__ import annotations

import logging
import sys

import structlog

from churnlens.config import settings


def _configure_structlog() -> None:
    """Configura structlog según los settings actuales."""
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        timestamper,
    ]

    if settings.log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


_configure_structlog()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Devuelve un logger estructurado.

    Args:
        name: nombre del logger (típicamente `__name__`).

    Returns:
        Un logger ya configurado, listo para usar.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
