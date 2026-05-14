# Criterios de éxito — ChurnLens

> **Propósito:** definir de manera explícita, medible y trazable los criterios que determinan si el proyecto es exitoso. Cada criterio se acompaña de su **umbral mínimo aceptable** y de la **fase TDSP** en la que se valida.

---

## 1. Criterios de negocio

| ID    | Criterio                                                                                  | Umbral mínimo            | Métrica                          | Validación |
|-------|-------------------------------------------------------------------------------------------|---------------------------|----------------------------------|------------|
| B-01  | _Lift_ en el decil superior frente a la base global                                       | ≥ **3.0 ×**              | `precision_top_decile / base_rate` | Fase 3     |
| B-02  | Captura del 50 % del _churn_ en menos del 20 % de la base                                 | ≤ **20 %** de la base    | `recall_at_50%_churn`            | Fase 3     |
| B-03  | ROI estimado de la campaña dirigida                                                       | > **0** (positivo)       | Ver `business_case.md`           | Fase 4     |
| B-04  | Reporte ejecutivo entregable a _stakeholders_ de negocio                                  | 1 documento + 1 deck     | Existencia + revisión            | Fase 5     |

---

## 2. Criterios técnicos (modelo)

| ID    | Criterio                                                          | Umbral mínimo  | Métrica                | Validación  |
|-------|-------------------------------------------------------------------|----------------|------------------------|-------------|
| T-01  | Calidad de _ranking_ (insensible al umbral)                       | ≥ **0.85**     | ROC-AUC                | Fase 3      |
| T-02  | Robustez al desbalance                                            | ≥ **0.65**     | PR-AUC                 | Fase 3      |
| T-03  | Equilibrio precision/recall sobre clase minoritaria               | ≥ **0.62**     | F1 (clase _Yes_)       | Fase 3      |
| T-04  | Sensibilidad sobre la clase minoritaria                           | ≥ **0.70**     | Recall (clase _Yes_)   | Fase 3      |
| T-05  | Calibración de probabilidades                                     | ≤ **0.05**     | ECE                    | Fase 3      |
| T-06  | Mejora frente a _baseline_ lineal                                 | ≥ **+3 pp**    | Δ F1                   | Fase 3      |
| T-07  | Reproducibilidad del entrenamiento                                | 100 %          | Re-run determinístico  | Fase 3      |

---

## 3. Criterios de ingeniería

| ID    | Criterio                                                          | Umbral mínimo            | Validación                  |
|-------|-------------------------------------------------------------------|--------------------------|-----------------------------|
| E-01  | Cobertura de tests sobre el paquete `churnlens`                   | ≥ **80 %**               | `pytest --cov`              |
| E-02  | _Type-checking_ estricto                                          | 0 errores                | `mypy --strict`             |
| E-03  | _Linting_                                                         | 0 errores                | `ruff check`                |
| E-04  | CI verde en `main`                                                | 100 % de los _runs_      | GitHub Actions              |
| E-05  | Validación de esquema sobre los datos crudos                      | Sin _failed checks_      | Pandera                     |
| E-06  | _Latency_ del API (Fase 4)                                        | < 200 ms p95             | _Load test_                 |
| E-07  | _Throughput_ del API (Fase 4)                                     | ≥ 100 RPS                | _Load test_                 |

---

## 4. Criterios de governance

| ID    | Criterio                                                          | Umbral / Esperado                            | Validación                |
|-------|-------------------------------------------------------------------|----------------------------------------------|---------------------------|
| G-01  | _Disparate Impact_ por género                                     | ∈ **[0.80, 1.25]**                           | Reporte de _fairness_     |
| G-02  | _Disparate Impact_ por `SeniorCitizen`                            | ∈ **[0.80, 1.25]**                           | Reporte de _fairness_     |
| G-03  | _Model Card_ completa                                             | Todas las secciones diligenciadas            | Revisión documental       |
| G-04  | Plan de monitoreo (Fase 4)                                        | PSI < 0.10 y alertas configuradas            | Documento de monitoreo    |
| G-05  | Atribución correcta del dataset                                   | Presente en `LICENSE` y `data_definition.md` | Revisión documental       |

---

## 5. Criterios de la entrega Fase 1 (rúbrica oficial)

> **Equivalente al 10 % de la nota — rango objetivo: 4.0 – 5.0.**

| ID    | Criterio rúbrica                                                                            | Estado |
|-------|---------------------------------------------------------------------------------------------|--------|
| F1-01 | Existe **marco del proyecto** (`project_charter.md`).                                       | ✅     |
| F1-02 | Existe **código de carga de datos** funcional y legible.                                    | ✅     |
| F1-03 | Existen **diccionarios de datos** (`data_dictionary.md` + `data_definition.md`).            | ✅     |
| F1-04 | El código de carga funciona (`make data` ejecuta exitosamente).                              | ✅     |
| F1-05 | Los documentos contienen información que corresponde a lo solicitado.                       | ✅     |

---

## 6. Reglas de aceptación final del proyecto

El proyecto se considera **aceptado** cuando:

1. Todos los criterios `B-*`, `T-*`, `E-*`, `G-*` están satisfechos sobre los umbrales mínimos.
2. La rúbrica oficial del curso califica cada fase ≥ 4.0 / 5.0.
3. El repositorio es **clonable y ejecutable end-to-end** por un tercero siguiendo el `README.md`.
4. La defensa final del proyecto se realiza con éxito ante el equipo evaluador.
