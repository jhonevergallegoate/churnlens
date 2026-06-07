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

---

## Fase 3 · Modelamiento y extracción de características · _10 %_

### 2026-05-27 — _Entrega Fase 3_

**Estado:** ✅ entregable completo.

**Hitos del día:**

- Implementación del módulo [`src/churnlens/features/selection.py`](../../src/churnlens/features/selection.py)
  con **cuatro técnicas complementarias** de selección y consolidación
  por consenso top-k:
  - Mutual Information (filtro univariado no lineal).
  - χ² (filtro categórico, omite numéricas escaladas).
  - L1 Logistic Regression (embedded, `class_weight=balanced`).
  - Permutation Importance sobre RandomForest (wrapper, scoring PR-AUC).
- Implementación del subpaquete [`src/churnlens/models/`](../../src/churnlens/models/):
  - `baseline.py` — Dummy {stratified, most_frequent, prior} + LogReg balanced.
  - `evaluation.py` — métricas (PR-AUC, ROC-AUC, F1, recall, precision,
    Brier, lift@10 %), _threshold tuning_ por F1, plots PR/ROC/CM/
    calibración/importancia/threshold-sweep.
  - `registry.py` — persistencia con manifest auditado (`hash_train`,
    `hash_val`, `hash_model`, métricas train/val/cv, hiperparámetros).
  - `train.py` — orquestador con catálogo de 8 modelos
    (`dummy_*`, `logreg_balanced`, `logreg_l1`, `random_forest`,
    `hist_gb`, `lightgbm`), CV estratificada 5-fold sobre PR-AUC y
    ROC-AUC, fit final en `train`, threshold tuning en `val`, save al
    registro.
- Extensión de la CLI con `churnlens features select`,
  `churnlens model train`, `churnlens model evaluate`, `churnlens model list`.
- Reemplazo de los _stubs_ [`scripts/training/main.py`](../../scripts/training/main.py) y
  [`scripts/evaluation/main.py`](../../scripts/evaluation/main.py)
  por orquestadores reales que reproducen la entrega byte-equivalente.
- Tres reportes Markdown nuevos en [`docs/modeling/`](../modeling/):
  - [`feature_selection.md`](../modeling/feature_selection.md) — justificación + top-20 consenso.
  - [`baseline_models.md`](../modeling/baseline_models.md) — comparativa CV + val (8 modelos).
  - [`final_model_report.md`](../modeling/final_model_report.md) — descripción detallada del ganador.
- Notebook narrativo [`notebooks/03_modeling_and_evaluation.ipynb`](../../notebooks/03_modeling_and_evaluation.ipynb).
- Suite de tests extendida (~30 tests nuevos) cubriendo selección,
  baseline, evaluación, registry y entrenamiento end-to-end con datos
  sintéticos.
- Pipeline de CI extendido con job `smoke-test-phase3` que entrena un
  subset rápido (baselines + LightGBM, CV 3-fold) y verifica todos los
  artefactos esperados.
- Dependencia nueva: `lightgbm>=4.3` añadida a `pyproject.toml`.

**Decisiones clave de la fase:**

- **Métrica primaria = PR-AUC**, no ROC-AUC, por el desbalance 26.5 %
  (la PR-AUC es más sensible al rendimiento sobre la clase positiva).
- **Threshold sintonizado maximizando F1 sobre `val`** — balance entre
  precision (no quemar inversión en retención) y recall (no perder
  cancelaciones).
- **`cross_val_score(n_jobs=1)`** dentro de la CV porque los modelos
  internos (RF, HGB, LightGBM) ya usan `n_jobs=-1`. Sin esto, se crean
  `n_cores²` workers y el sistema entra en _live-lock_ con LightGBM.
- **Modelo final = `logreg_l1`** por principio de parsimonia: empate
  estadístico con `logreg_balanced` (Δ ≤ 0.0001 PR-AUC en CV y val), 8
  de 35 features apagadas automáticamente por la regularización L1,
  interpretabilidad equivalente.
- **El consenso top-20 se calcula pero no se aplica al ganador** porque
  L1 ya implementa selección embedded y reproduce el mismo subset. El
  manifest queda disponible para futuros modelos que no esparsen
  naturalmente (RF, GBM).
- **`test.parquet` queda intocado** — se evalúa solo en Fase 4 para
  evitar _data leakage_ por _threshold tuning_ implícito.

**Resultados clave (números reales sobre `val`):**

| Modelo            | PR-AUC val | ROC-AUC val | F1 tuned | thr |
|-------------------|-----------:|------------:|---------:|----:|
| **`logreg_l1`** ★ | **0.6293** | **0.8286**  | **0.6390** | 0.58 |
| `logreg_balanced` | 0.6290     | 0.8286      | 0.6379   | 0.58 |
| `random_forest`   | 0.6131     | 0.8241      | 0.6154   | 0.27 |
| `lightgbm`        | 0.6113     | 0.8176      | 0.6071   | 0.35 |
| `hist_gb`         | 0.6103     | 0.8208      | 0.6091   | 0.48 |
| `dummy_prior`     | 0.2658     | 0.5000      | 0.4200   | 0.05 |

- Lift @ top 10 % del ganador: **2.70 ×** vs base rate 0.266.
- Mejor modelo no lineal (RF) **no supera** al baseline lineal real.
- Top-5 coeficientes coinciden con top-5 consenso de selección.

**Workflow operativo introducido en esta fase:**

- Rama `dev` para desarrollo libre, `main` protegida con branch
  protection (requiere PR, sin force push, sin deletion).
- Cuenta gh activa para el proyecto: `jhonevergallegoate` con email
  `jhgallegoa21@gmail.com` (separación profesional/académico
  preservada).

### Próximos pasos (Fase 4)

- Evaluación sobre `test.parquet` y reporte definitivo held-out.
- Calibración isotónica si Brier > 0.18 en test.
- _Fairness audit_ por `gender` y `SeniorCitizen` (paridad de recall y
  precision por subgrupo).
- API REST + monitoreo de _data drift_, alertas por degradación de PR-AUC.
- Plan de re-entrenamiento periódico.

---

## Fase 4 · Despliegue · _10 %_

### 2026-06-03 — _Entrega Fase 4_

**Estado:** ✅ entregable completo.

**Hitos del día:**

- Implementación del subpaquete [`src/churnlens/serving/`](../../src/churnlens/serving/):
  - `schemas.py` — contratos Pydantic de entrada/salida, espejo exacto del
    [data dictionary](../data/data_dictionary.md) (dominios cerrados con
    `Literal`, rangos, `extra="forbid"`) **+ las dos reglas de integridad
    cruzada del esquema Pandera replicadas como validadores** (payloads
    fuera de contrato → `422`).
  - `service.py` — `ChurnScorer`: pipeline de inferencia completo (features
    derivadas → `ColumnTransformer` de Fase 2 → `logreg_l1` → threshold
    0.58), con carga única de artefactos en el startup.
  - `api.py` — FastAPI con `GET /health`, `GET /metadata`,
    `POST /predict`, `POST /predict/batch` (≤ 1 000 clientes), middleware
    de latencia (`X-Process-Time-Ms`), OpenAPI en `/docs`.
- Extensión de la CLI con `churnlens serve` (uvicorn con host/port/workers
  configurables vía `Settings`).
- Script TDSP oficial [`scripts/deployment/main.py`](../../scripts/deployment/main.py):
  smoke E2E in-process de los 4 endpoints (+ caso 422), con evidencia
  persistida en `reports/tables/deployment_smoke.json` y modo
  `--ensure-artifacts` que reconstruye datos + preprocesador + modelo.
- [`Dockerfile`](../../Dockerfile) **multi-stage reproducible**: la etapa
  builder descarga el dataset (checksum), ajusta el preprocesador y entrena
  `logreg_l1` desde cero (semilla 42); la etapa runtime es
  `python:3.12-slim` + usuario no-root + `HEALTHCHECK` + uvicorn ×2 workers.
  Verificado: la imagen reconstruida produce **probabilidades
  byte-equivalentes** a las del entrenamiento local (p = 0.868361 para el
  payload de referencia).
- [`docker-compose.yml`](../../docker-compose.yml) con healthcheck, límites
  de recursos (1 vCPU / 512 MB) y notas de escalado horizontal.
- Dos documentos nuevos en [`docs/deployment/`](../deployment/):
  - [`deploymentdoc.md`](../deployment/deploymentdoc.md) — estructura del
    template TDSP: infraestructura, código, instalación, configuración,
    uso, mantenimiento + validación del despliegue.
  - [`infrastructure.md`](../deployment/infrastructure.md) — componentes,
    dimensionamiento, **comparativa de plataformas con costos** (Cloud Run
    recomendada: ~$0–5/mes con scale-to-zero) y **plan de
    mantenimiento/monitoreo** (PSI, latencia, runbook de re-entrenamiento,
    ~60 h/año de operación).
- Suite de tests extendida: **31 tests nuevos** de serving (scorer, bandas
  de riesgo, contratos, endpoints, batch) sobre artefactos sintéticos de
  sesión — total **128 tests**, cobertura 86 %.
- CI extendido con dos jobs: `smoke-test-phase4` (smoke in-process con
  verificación del JSON de evidencia) y `docker-smoke` (build de la imagen
  + `/health` + `/predict` sobre HTTP real).
- Targets de Makefile: `serve`, `deploy-smoke`, `docker-build`,
  `docker-up`, `docker-down`, `phase4`.
- Versión del paquete: `0.2.0` → `0.3.0`; dependencias nuevas: `fastapi`,
  `uvicorn[standard]` (core) y `httpx` (dev).

**Decisiones clave de la fase:**

- **API REST sobre FastAPI** (no batch-only): es la forma de "puesta en
  producción" que el CRM/equipo de retención puede consumir, cumple
  "eficiente y escalable" de la rúbrica y era lo comprometido en la
  arquitectura desde Fase 1.
- **Artefactos horneados en la imagen** (no volúmenes): el `docker build`
  reconstruye el pipeline completo desde un checkout limpio — la imagen es
  autocontenida, inmutable y versionable; el rollback es volver al tag
  anterior.
- **Threshold operable por entorno** (`CHURNLENS_SERVING_THRESHOLD`) sin
  rebuild — default: el 0.58 sintonizado del manifiesto.
- **Carga perezosa (PEP 562) de `churnlens.models.train`** en el
  `__init__` del subpaquete: el serving ya no arrastra LightGBM/libgomp a
  la imagen de producción.
- **Endpoints síncronos (`def`)**: FastAPI los despacha al threadpool — el
  trabajo CPU-bound de sklearn no bloquea el event loop.
- **`populate_by_name=True` en `Settings`**: las rutas (`data_dir`,
  `models_dir`) ahora tienen alias de entorno (`CHURNLENS_DATA_DIR`,
  `CHURNLENS_MODELS_DIR`) para redirigirse dentro del contenedor sin
  romper la construcción por nombre de campo usada en tests.

**Resultados clave (held-out `test.parquet`, primera apertura — threshold fijo 0.58):**

| Métrica | `val` (Fase 3) | `test` (Fase 4) |
|---------|---------------:|----------------:|
| PR-AUC | 0.6293 | **0.6313** |
| ROC-AUC | 0.8286 | **0.8460** |
| F1 | 0.6390 | 0.6145 |
| Recall | 0.7402 | 0.7286 |
| Brier | 0.1700 | 0.1678 |
| Lift @ 10 % | 2.70× | **2.78×** |

Sin degradación material — el modelo desplegado generaliza y el threshold
sintonizado en `val` se sostiene.

### Próximos pasos (Fase 5 — entrega final)

- _Fairness audit_ por `gender` y `SeniorCitizen` (pendiente de Fase 3).
- Completar la [model card](../governance/model_card.md) con los números
  finales de test.
- Demo end-to-end + reporte final consolidado de las 5 fases.
- (Opcional) despliegue real en Cloud Run con autenticación IAM.

---

## Fase 5 · Evaluación final + Aceptación · _entrega final_

### 2026-06-07 — _Entrega Fase 5 (cierre del proyecto)_

**Estado:** ✅ entregable completo — **proyecto cerrado, versión `1.0.0`**.

**Hitos del día:**

- Implementación del módulo [`src/churnlens/models/fairness.py`](../../src/churnlens/models/fairness.py):
  - Métricas por subgrupo (`selection_rate`, TPR, FPR, precision, ECE
    con bins uniformes) y consolidación por atributo (**Disparate
    Impact**, **Demographic Parity diff**, **Equalized Odds diff**,
    max ECE) contra los umbrales declarados en
    [`ethics_and_fairness.md` §3](../governance/ethics_and_fairness.md).
  - Reconstrucción de los atributos sensibles crudos **replicando el
    split determinista de la Fase 2** (semilla 42), con verificación de
    alineación contra la secuencia completa del target de
    `test.parquet` (divergencia → `RuntimeError`).
  - Persistencia de evidencia: `fairness_groups_logreg_l1.csv`,
    `fairness_summary_logreg_l1.json` y figura comparativa
    `fairness_audit_logreg_l1.png`.
- Extensión de la CLI con `churnlens model fairness` y script TDSP
  oficial [`scripts/evaluation/fairness_audit.py`](../../scripts/evaluation/fairness_audit.py)
  (modo `--strict` para uso como gate de CI).
- Targets de Makefile: `fairness`, `phase5`.
- CI extendido con el job `smoke-test-phase5` (pipeline real → audit →
  verificación estructural del JSON + paridad de género).
- Suite de tests: **14 tests nuevos** de fairness (unitarios exactos +
  integración end-to-end sobre artefactos sintéticos) — total
  **142 tests**, cobertura 85 %.
- [Model card](../governance/model_card.md) **completa**: identidad
  final del modelo, métricas train/CV/test contra umbrales objetivo,
  partición definitiva, auditoría cuantitativa de fairness (§7) e
  interpretación.
- [`ethics_and_fairness.md`](../governance/ethics_and_fairness.md)
  actualizado con resultados (§3.1) y **decisiones de mitigación
  documentadas** (§4).
- Nuevo entregable de aceptación
  [`docs/acceptance/exit_report.md`](../acceptance/exit_report.md)
  (informe de salida TDSP): resultados por fase, evaluación final vs
  baseline, cumplimiento de criterios de éxito, lecciones aprendidas,
  impacto y conclusiones.
- Versión del paquete: `0.3.0` → **`1.0.0`** (Development Status:
  Production/Stable).

**Decisiones clave de la fase:**

- **La auditoría reconstruye los atributos sensibles desde el crudo**
  en lugar de propagarlos por el parquet transformado: no contamina el
  feature set del modelo y aprovecha el determinismo del split. La
  alineación se verifica, no se asume.
- **`SeniorCitizen` se audita aunque no es feature** — quedó excluido
  del `ColumnTransformer` (ningún bloque lo lista y `remainder="drop"`
  lo descarta); el riesgo de discriminación vía _proxies_ se evalúa
  igual. El hallazgo de la exclusión silenciosa queda como lección
  aprendida en el informe de salida.
- **No se aplica reponderación pese a DI < 0.80** en
  `SeniorCitizen`/`Partner`/`Dependents` — desviación documentada: las
  disparidades reflejan prevalencias reales (seniors churnean 44.4 % vs
  23.1 %) y la acción del modelo es beneficiosa (ofertas de retención);
  igualar tasas de selección redirigiría beneficios fuera de los grupos
  que más churnean (Kleinberg et al., 2016).
- **El ECE > 0.05 se reporta como global, no diferencial**
  (test completo 0.153; subgrupos 0.105-0.175): proviene de
  `class_weight=balanced`. Mitigación (recalibración isotónica) queda
  como primer ítem de v1.1, no bloquea el release.
- **La auditoría es informativa, no bloqueante** en CI (el job verifica
  estructura y paridad de género; el modo `--strict` queda disponible).

**Resultados clave (fairness · test · threshold 0.58):**

| Atributo        | DI    | DPD   | EOD   | max ECE | Veredicto |
|-----------------|------:|------:|------:|--------:|-----------|
| `gender`        | 0.998 | 0.001 | 0.012 | 0.156   | ✓ paridad (solo falla ECE global) |
| `SeniorCitizen` | 0.519 | 0.293 | 0.245 | 0.172   | ⚠ prevalencia-driven |
| `Partner`       | 0.480 | 0.250 | 0.195 | 0.173   | ⚠ prevalencia-driven |
| `Dependents`    | 0.387 | 0.276 | 0.224 | 0.175   | ⚠ prevalencia-driven |

- Riesgo residual bajo monitoreo: brecha de TPR ~0.20 en
  `Partner`/`Dependents` (churners con pareja/dependientes se detectan
  menos).

### Cierre

Con esta entrega el proyecto queda **completo en sus cinco fases**:
negocio → datos → modelado → despliegue → evaluación final y
aceptación. La evidencia consolidada vive en el
[informe de salida](../acceptance/exit_report.md).
