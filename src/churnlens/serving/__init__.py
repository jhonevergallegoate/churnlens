"""Capa de despliegue del modelo (Fase 4).

Expone el modelo ganador de la Fase 3 como un servicio HTTP de inferencia:

* :mod:`churnlens.serving.schemas` — contratos Pydantic de entrada/salida.
* :mod:`churnlens.serving.service` — :class:`ChurnScorer`, el pipeline de
  inferencia (features derivadas → preprocesador → modelo → decisión).
* :mod:`churnlens.serving.api` — aplicación FastAPI con los endpoints
  ``/health``, ``/metadata``, ``/predict`` y ``/predict/batch``.
"""

from __future__ import annotations
