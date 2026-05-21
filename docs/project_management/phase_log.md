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

---

## Fase 2 · Preprocesamiento + Análisis Exploratorio · _10 %_

### 2026-05-20 — _Entrega Fase 2_

**Estado:** ✅ entregable completo.

**Hitos del día:**

- Implementación del subpaquete `churnlens.features`:
  - `engineering.py` — 7 features derivadas (`tenure_bucket`, `services_count`,
    `has_internet`, `has_phone`, `auto_payment`, `avg_monthly_spend`, `monthly_spend_gap`).
  - `preprocessing.py` — `ColumnTransformer` con bloques numérico
    (imputación mediana + StandardScaler), ordinal (`Contract`,
    `tenure_bucket`), binario y nominal (`OneHotEncoder(drop="first")`).
  - `splits.py` — partición estratificada 70/15/15 con `random_state=42`.
  - `pipeline.py` — orquestador end-to-end que materializa
    `data/processed/*.parquet`, `preprocessor.joblib`, `feature_names.json`
    y `metadata.json`.
- Implementación del subpaquete `churnlens.eda`:
  - `summary.py` — estadísticas descriptivas numéricas / categóricas,
    distribución del target, V de Cramér corregida y matriz de
    correlación Spearman.
  - `plots.py` — 6 funciones de visualización reutilizables
    (histogramas, boxplots, churn rate por categoría, heatmap, missing).
  - `report.py` — genera 9 figuras PNG + 4 tablas CSV reproducibles.
- Extensión de la CLI con `churnlens preprocess run` y `churnlens eda report`.
- Reemplazo de los stubs `scripts/preprocessing/main.py` y `scripts/eda/main.py`
  por implementaciones reales.
- Nuevo entregable [`docs/data/data_summary_report.md`](../data/data_summary_report.md)
  con 13 secciones (resumen ejecutivo, estructura, target, descriptivas,
  bivariado, correlaciones, faltantes, decisiones, artefactos,
  reproducibilidad y conclusiones).
- Extensión de [`data_dictionary.md`](../data/data_dictionary.md) (§6 features derivadas, §7 artefactos)
  y de [`data_definition.md`](../data/data_definition.md) (§3.3 archivos Fase 2, §5.4 pipeline,
  referencias internas actualizadas).
- Notebook [`notebooks/02_eda_and_preprocessing.ipynb`](../../notebooks/02_eda_and_preprocessing.ipynb)
  con narrativa exploratoria + ejecución del pipeline.
- Suite de tests extendida para cubrir `features.engineering`,
  `features.preprocessing`, `features.splits`, `features.pipeline` y
  `eda.summary`.

**Decisiones clave de la fase:**

- **Imputación dentro del `ColumnTransformer`** (mediana sobre `train`),
  no antes del split, para evitar _leakage_.
- **No descartar `gender` ni `PhoneService`** en Fase 2 pese a V Cramér
  ≈ 0; la selección de features cuantitativa se hace en Fase 3.
- **Conservar `MonthlyCharges` y `avg_monthly_spend`** pese a su
  correlación de Spearman = 0.99 — útil para modelos basados en árboles
  y para separar cambios de plan en modelos lineales con regularización.
- **Split 70/15/15** estratificado con semilla fija 42 — tasa de churn
  preservada en los tres conjuntos (26.5 % ± 0.1 pp).

**Hallazgos del EDA (números reales):**

- Tasa global de churn: **26.54 %** (1 869 / 7 043).
- Predictor más fuerte: `Contract` (V Cramér = **0.410**).
  Month-to-month churnea **42.7 %**, two-year churnea **2.8 %**.
- `tenure_bucket` 0-12m churnea **47.4 %** vs 49-72m **9.5 %** (5× menos).
- `PaymentMethod = Electronic check` churnea **45.3 %** vs métodos
  automáticos (15-17 %).
- `gender` y `PhoneService` no muestran asociación con churn
  (V Cramér ≈ 0).
- 11 NaN en `TotalCharges` — todos con `tenure = 0`.

### Próximos pasos (Fase 3)

- Selección de features cuantitativa (información mutua, permutation importance).
- Benchmark de modelos lineales + árboles + GBM con validación cruzada estratificada.
- Calibración del _threshold_ para la métrica de negocio.
- Reporte de fairness por subgrupos (gender, SeniorCitizen).
