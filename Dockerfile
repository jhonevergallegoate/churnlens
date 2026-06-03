# syntax=docker/dockerfile:1
# =====================================================================
# ChurnLens — imagen de producción de la API de inferencia (Fase 4)
#
# Build multi-stage:
#   1. builder  — instala el paquete y RECONSTRUYE los artefactos de
#                 inferencia desde cero (descarga validada → preprocesador
#                 → modelo ganador). La imagen no depende de artefactos
#                 locales no versionados: `docker build` es reproducible
#                 desde un checkout limpio del repositorio.
#   2. runtime  — imagen mínima: venv + modelo + preprocesador, usuario
#                 no-root, HEALTHCHECK y uvicorn con 2 workers.
#
# Uso:
#   docker build -t churnlens-api .
#   docker run --rm -p 8000:8000 churnlens-api
# =====================================================================

# ---------------------------------------------------------------------
# Etapa 1 — builder
# ---------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Los scripts del builder escriben en /build, no en site-packages.
    CHURNLENS_DATA_DIR=/build/data \
    CHURNLENS_MODELS_DIR=/build/models

# libgomp1: runtime de OpenMP requerido por LightGBM durante el entrenamiento.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instala primero el paquete (capa cacheable mientras no cambie el código).
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --upgrade pip && pip install .

# Reconstruye los artefactos de inferencia:
#   Fase 1: descarga + validación Pandera + checksums.
#   Fase 2: features derivadas + ColumnTransformer ajustado + splits.
#   Fase 3: entrenamiento del modelo ganador (logreg_l1, seed fija = 42).
COPY scripts/ scripts/
RUN python scripts/data_acquisition/main.py --quiet \
 && python scripts/preprocessing/main.py --quiet \
 && python scripts/training/main.py --quiet --skip-selection --cv 3 --models logreg_l1

# ---------------------------------------------------------------------
# Etapa 2 — runtime
# ---------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="ChurnLens API" \
      org.opencontainers.image.description="Servicio de inferencia de churn (Diplomado MLDS · UNAL)" \
      org.opencontainers.image.source="https://github.com/jhonevergallegoate/churnlens" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    CHURNLENS_DATA_DIR=/app/data \
    CHURNLENS_MODELS_DIR=/app/models

RUN useradd --create-home --uid 1000 churnlens

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Solo los artefactos que la API necesita en runtime (≈ 15 KB).
COPY --from=builder /build/models/logreg_l1.joblib        models/
COPY --from=builder /build/models/logreg_l1.metadata.json models/
COPY --from=builder /build/data/processed/preprocessor.joblib data/processed/
COPY --from=builder /build/data/processed/feature_names.json  data/processed/
COPY --from=builder /build/data/processed/metadata.json       data/processed/

USER churnlens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "churnlens.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
