# =====================================================================
# ChurnLens — Makefile
# Comandos cortos y reproducibles para todo el ciclo de vida del proyecto.
# =====================================================================

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

PYTHON       ?= python
PIP          ?= $(PYTHON) -m pip
PROJECT_NAME := churnlens
SRC_DIR      := src/$(PROJECT_NAME)
TESTS_DIR    := tests
SCRIPTS_DIR  := scripts
DATA_DIR     := data

# -- Colores para mejorar la lectura del help -------------------------
BLUE   := \033[1;34m
GREEN  := \033[1;32m
YELLOW := \033[1;33m
RESET  := \033[0m

.DEFAULT_GOAL := help

# =====================================================================
# Help
# =====================================================================
.PHONY: help
help:  ## Muestra esta ayuda
	@printf "$(BLUE)ChurnLens$(RESET) — comandos disponibles:\n\n"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n"

# =====================================================================
# Entorno
# =====================================================================
.PHONY: install install-dev clean-venv
install:  ## Instala el paquete en modo editable
	$(PIP) install -U pip
	$(PIP) install -e .

install-dev:  ## Instala el paquete con extras de desarrollo y notebooks
	$(PIP) install -U pip
	$(PIP) install -e ".[all]"
	pre-commit install || true

clean-venv:  ## Elimina caches de herramientas (no toca .venv)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# =====================================================================
# Datos (Fase 1)
# =====================================================================
.PHONY: data data-download data-validate data-summary data-clean
data: data-download data-validate data-summary  ## Pipeline completo de la Fase 1 (descarga + validación + perfil)

data-download:  ## Descarga el dataset crudo a data/raw/
	$(PYTHON) -m churnlens.cli data download

data-validate:  ## Valida el dataset crudo contra el esquema Pandera
	$(PYTHON) -m churnlens.cli data validate

data-summary:  ## Imprime un perfil rápido del dataset
	$(PYTHON) -m churnlens.cli data summary

data-clean:  ## Elimina datos descargados (mantiene .gitkeep)
	find $(DATA_DIR) -type f ! -name '.gitkeep' ! -name 'README.md' -delete

# =====================================================================
# Calidad
# =====================================================================
.PHONY: lint format type-check test test-cov check
lint:  ## Ejecuta ruff
	ruff check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

format:  ## Aplica formato con ruff y black
	ruff check --fix $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	ruff format $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	black $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

type-check:  ## Ejecuta mypy estricto sobre src/
	mypy $(SRC_DIR)

test:  ## Ejecuta los tests con pytest
	pytest

test-cov:  ## Ejecuta los tests con reporte de cobertura HTML
	pytest --cov-report=html

check: lint type-check test  ## Ejecuta lint + type-check + tests

# =====================================================================
# Notebooks
# =====================================================================
.PHONY: notebook nb-strip
notebook:  ## Lanza Jupyter Lab
	jupyter lab notebooks/

nb-strip:  ## Limpia outputs de los notebooks antes de commitear
	nbstripout notebooks/*.ipynb

# =====================================================================
# Documentación
# =====================================================================
.PHONY: tree
tree:  ## Imprime el árbol del proyecto (excluye datos, venv, caches)
	@tree -a -I '__pycache__|.venv|.git|.pytest_cache|.ruff_cache|.mypy_cache|data|.ipynb_checkpoints|*.egg-info|htmlcov' || \
	find . -maxdepth 3 -not -path '*/.*' -print
