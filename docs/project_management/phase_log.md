# Bitácora del proyecto — ChurnLens

> Registro cronológico de hitos, decisiones y entregas por fase TDSP.

---

## Fase 1 · Entendimiento del negocio + Carga de datos · _10 %_

### 2026-05-14 — _Kickoff y entrega Fase 1_

**Estado:** ✅ entregable completo.

**Hitos del día:**

- Inicialización del repositorio con estructura TDSP extendida.
- Redacción del [Project Charter](../business_understanding/project_charter.md) completo (18 secciones, objetivos SMART, métricas técnicas + de negocio).
- Redacción de documentación complementaria: [business case](../business_understanding/business_case.md), [stakeholders](../business_understanding/stakeholders.md), [success criteria](../business_understanding/success_criteria.md), [glossary](../business_understanding/glossary.md).
- Redacción del [diccionario de datos](../data/data_dictionary.md) cubriendo las 21 variables del dataset _Telco Customer Churn_.
- Redacción del [data definition](../data/data_definition.md) con orígenes, rutas, transformaciones, licencia.
- Redacción del [reporte preliminar de calidad](../data/data_quality_report.md).
- Implementación del paquete `churnlens` (Pydantic config, structlog logger, Typer CLI, Pandera schema, loader, validators, hashing utils).
- Implementación del script TDSP `scripts/data_acquisition/main.py`.
- Suite de tests con **26 tests pasando** y **68 %** de cobertura global (74-100 % en módulos críticos).
- Descarga real del dataset desde el _mirror_ oficial de IBM verificada end-to-end:
  - 7 043 filas × 21 columnas.
  - Tasa de churn: **26.54 %**.
  - MD5: `3b0bfab28a8101b4e4fdd08025a5c235`.
- Documentos de _governance_: [ethics & fairness](../governance/ethics_and_fairness.md), [privacy & compliance](../governance/privacy_and_compliance.md), [model card](../governance/model_card.md) (template).
- [Arquitectura de la solución](../architecture/solution_architecture.md) con diagrama Mermaid.
- Configuración de CI con GitHub Actions (lint + types + tests + smoke-test del pipeline).
- Pre-commit hooks configurados.

**Decisiones clave:**

- _Caso de uso_: predicción temprana de _churn_ en servicios por suscripción, framing genérico (sin mención a empresas específicas).
- _Template TDSP_: usado como guía para asegurar que los evaluadores encuentren los entregables donde esperan, **extendido** con artefactos adicionales de _governance_, arquitectura y CI.
- _Repo_: privado en GitHub personal del autor.
- _Equipo_: individual (dentro del límite de 3 personas permitido por la rúbrica).
- _Stack_: Python 3.10+, Pandas, Pandera, Pydantic, Typer, structlog, pytest.

### Próximos pasos (Fase 2)

- EDA reproducible con `ydata-profiling` + visualizaciones específicas.
- Decisión definitiva sobre imputación de `TotalCharges`.
- Pipeline de preprocesamiento (`ColumnTransformer`).
- Diseño y benchmark de _feature engineering_ (`tenure_bucket`, `services_count`, etc.).
