## Resumen

<!-- Describe brevemente qué se incluye en este PR (1-3 líneas). -->

## Tipo de cambio

- [ ] Documentación (TDSP / READMEs)
- [ ] Código de paquete (`src/churnlens`)
- [ ] Scripts TDSP (`scripts/*`)
- [ ] Tests
- [ ] Configuración (CI, lint, pre-commit, etc.)
- [ ] Otro: <!-- describir -->

## Fase TDSP

- [ ] Fase 1 · Entendimiento del negocio + carga
- [ ] Fase 2 · EDA + preprocesamiento
- [ ] Fase 3 · Modelado
- [ ] Fase 4 · Despliegue
- [ ] Fase 5 · Evaluación / entrega final

## Checklist

- [ ] El código pasa `ruff check` y `ruff format --check`.
- [ ] El código pasa `mypy --strict`.
- [ ] Los tests pasan localmente (`pytest`).
- [ ] La cobertura no decrece.
- [ ] La documentación afectada está actualizada.
- [ ] Si toca el esquema de datos: `data_dictionary.md` y `data_definition.md` están alineados.
- [ ] Si introduce dependencias nuevas: están registradas en `pyproject.toml`.
