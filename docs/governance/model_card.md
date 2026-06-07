# Model Card — ChurnLens

> **Estado en la Fase 5 (final):** _completa._ Métricas held-out, partición definitiva y auditoría cuantitativa de _fairness_ diligenciadas con los resultados reales del modelo desplegado. Estructura basada en _Mitchell et al., 2019 — Model Cards for Model Reporting_.

---

## 1. Identidad del modelo

| Campo                       | Valor                                                                                |
|-----------------------------|--------------------------------------------------------------------------------------|
| Nombre del modelo           | `logreg_l1` — Regresión logística con regularización L1 (`C=0.5`, `class_weight=balanced`, `solver=liblinear`) |
| Versión                     | `1.0.0` (release final · Fase 5)                                                      |
| Tipo de problema            | Clasificación binaria supervisada                                                     |
| Variable objetivo           | `Churn ∈ {Yes, No}`                                                                   |
| Dominio                     | Servicios por suscripción (telco / SaaS / fintech / streaming, etc.)                  |
| Fecha de _release_           | 2026-06-06                                                                            |
| Autor / Propietario         | Jhon Gallego                                                                          |
| Licencia                    | MIT                                                                                   |
| Repositorio                 | `https://github.com/jhonevergallegoate/churnlens`                                      |
| Manifiesto auditado         | [`models/logreg_l1.metadata.json`](../../models/) (hashes SHA-256 de datos y modelo, hiperparámetros, métricas) |
| Threshold operativo         | **0.58** (sintonizado por F1 sobre `val`; operable vía `CHURNLENS_SERVING_THRESHOLD`) |

El modelo consume **35 features** post-`ColumnTransformer` (Fase 2); la regularización L1 apaga 8 de ellas automáticamente. Fue seleccionado por **principio de parsimonia**: empate estadístico con `logreg_balanced` (Δ ≤ 0.0001 PR-AUC) y superior a Random Forest, HistGradientBoosting y LightGBM tanto en CV como en validación (ver [`baseline_models.md`](../modeling/baseline_models.md)).

---

## 2. Uso previsto

### 2.1 Casos de uso primarios

- Priorizar contactos del equipo de retención sobre clientes con alta probabilidad de cancelación voluntaria en el próximo ciclo de facturación.
- Insumo para diseñar campañas de retención **dirigidas** y medir su _uplift_.
- Material académico para enseñar metodologías TDSP, MLOps ligero y _fairness in ML_.

### 2.2 Usuarios previstos

- Equipo de Retención (consumidor de la lista priorizada).
- Equipo de Producto (consumidor de los hallazgos sobre _drivers_).
- Evaluadores académicos del Diplomado MLDS (UNAL).

### 2.3 Casos de uso **fuera del alcance**

- Denegación de servicio o cobro diferencial a clientes individuales.
- Decisiones automatizadas sin revisión humana.
- Aplicación a poblaciones distintas a las representadas por el dataset (extrapolación injustificada).
- Combinación con datos personales identificables sin un nuevo proceso de evaluación.

---

## 3. Factores

### 3.1 Factores relevantes

- `Contract`, `tenure`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, _add-ons_ del servicio. Los coeficientes top-5 del modelo coinciden con el top-5 del consenso de selección de features (ver [`feature_selection.md`](../modeling/feature_selection.md)).

### 3.2 Atributos sensibles evaluados

- `gender`, `Partner`, `Dependents` (entran como features) y `SeniorCitizen` (**no** entra como feature — el `ColumnTransformer` lo descarta — pero se audita igualmente por posible discriminación vía _proxies_). Resultados en §7.

---

## 4. Métricas

Métricas del modelo final sobre los tres contextos de evaluación. `train` a threshold 0.5; `test` (held-out, primera y única apertura en Fase 4) a threshold operativo 0.58. La CV es estratificada 5-fold sobre `train` y solo reporta métricas de _ranking_ (no dependen de threshold). Umbrales objetivo definidos en la Fase 1 ([`success_criteria.md`](../business_understanding/success_criteria.md)).

| Métrica            | Train  | CV (k=5)          | Test (held-out) | Umbral objetivo | Cumple |
|--------------------|:------:|:-----------------:|:---------------:|:---------------:|:------:|
| ROC-AUC            | 0.8506 | 0.8466 ± 0.0040   | **0.8460**      | ≥ 0.85          | ✗ (−0.4 pp) |
| PR-AUC             | 0.6678 | 0.6613 ± 0.0173   | **0.6313**      | ≥ 0.65          | ✗ (−1.9 pp) |
| F1 (clase Yes)     | 0.6344 | —                 | **0.6145**      | ≥ 0.62          | ✗ (−0.6 pp) |
| Recall (clase Yes) | 0.8112 | —                 | **0.7286**      | ≥ 0.70          | ✓      |
| ECE (calibración)  | —      | —                 | **0.1527**      | ≤ 0.05          | ✗      |
| Lift top decil     | 2.88×  | —                 | **2.78×**       | ≥ 3.0×          | ✗ (marginal) |

**Lectura honesta de los umbrales.** Los objetivos de la Fase 1 resultaron ambiciosos para este dataset (~7 K filas, señal limitada a variables estructurales). Las métricas de _ranking_ quedan a décimas del objetivo y el modelo **sí supera ampliamente el criterio de utilidad**: PR-AUC 2.4× sobre el baseline trivial (0.6313 vs 0.2649) y lift@10 % de 2.78× — el equipo de retención que contacte el decil superior encuentra 2.78 veces más churners reales que contactando al azar. La brecha material es **calibración**: `class_weight=balanced` infla sistemáticamente las probabilidades (media predicha 0.42 vs tasa real 0.26); el threshold 0.58 compensa la decisión binaria, pero las probabilidades crudas **no deben leerse como frecuencias** sin recalibración isotónica (trabajo futuro documentado en §9).

Sin degradación material entre `val` y `test` (PR-AUC 0.6293 → 0.6313; ROC-AUC 0.8286 → 0.8460): el modelo generaliza y el threshold sintonizado en `val` se sostiene en producción.

---

## 5. Datos de entrenamiento

| Campo                       | Valor                                                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------------------------------|
| Dataset                     | _Telco Customer Churn_ (IBM Sample Data Sets).                                                                  |
| Tamaño                      | 7 043 filas × 21 columnas (MD5 verificado en la descarga).                                                      |
| Partición                   | **70/15/15** estratificada por `Churn`, semilla 42 → `train` 4 929 · `val` 1 057 · `test` 1 057.                 |
| Tasa de positivos           | 26.54 % global (26.5 % ± 0.1 pp preservada en los tres splits).                                                 |
| Features                    | 13 originales + 7 derivadas → 35 columnas post-transformación (ver [`data_dictionary.md`](../data/data_dictionary.md) §6-7). |
| Documentación               | [`data_dictionary.md`](../data/data_dictionary.md) · [`data_definition.md`](../data/data_definition.md) · [`data_quality_report.md`](../data/data_quality_report.md). |

---

## 6. Datos de evaluación

- **Held-out:** `test.parquet` — 1 057 filas, tasa de positivos 26.49 %, **nunca tocado** durante selección de features, entrenamiento ni _threshold tuning_ (Fases 2-3). Primera apertura al cierre de la Fase 4, con el threshold ya congelado.
- **Validación:** `val.parquet` — 1 057 filas; usado para threshold tuning y selección del modelo ganador.
- La evaluación es **reproducible byte a byte**: `make phase3 && make evaluate` regenera las métricas; el manifiesto del modelo registra los hashes SHA-256 de los datos usados.

---

## 7. Análisis cuantitativo de _fairness_

Auditoría sobre `test` (n = 1 057) con threshold 0.58, ejecutada con [`scripts/evaluation/fairness_audit.py`](../../scripts/evaluation/fairness_audit.py) (módulo [`churnlens.models.fairness`](../../src/churnlens/models/fairness.py)). Metodología y umbrales en [`ethics_and_fairness.md`](ethics_and_fairness.md) §3. Evidencia reproducible con `make fairness` (también artifact del CI): `fairness_groups_logreg_l1.csv` y `fairness_summary_logreg_l1.json` en `reports/tables/`, figura en `reports/figures/fairness_audit_logreg_l1.png`.

### 7.1 Métricas por subgrupo

| Subgrupo                | n   | Prevalencia | Tasa de selección | TPR (recall) | FPR    | Precision |
|-------------------------|----:|:-----------:|:-----------------:|:------------:|:------:|:---------:|
| gender = Female         | 529 | 27.0 %      | 36.3 %            | 0.734        | 0.225  | 0.547     |
| gender = Male           | 528 | 26.0 %      | 36.4 %            | 0.723        | 0.238  | 0.516     |
| SeniorCitizen = 0       | 888 | 23.1 %      | 31.6 %            | 0.698        | 0.202  | 0.509     |
| SeniorCitizen = 1       | 169 | 44.4 %      | 61.0 %            | 0.813        | 0.447  | 0.592     |
| Partner = No            | 556 | 33.3 %      | 48.2 %            | 0.795        | 0.326  | 0.549     |
| Partner = Yes           | 501 | 19.0 %      | 23.2 %            | 0.600        | 0.145  | 0.491     |
| Dependents = No         | 724 | 31.4 %      | 45.0 %            | 0.771        | 0.304  | 0.537     |
| Dependents = Yes        | 333 | 15.9 %      | 17.4 %            | 0.547        | 0.104  | 0.500     |

### 7.2 Indicadores agregados por atributo

| Atributo        | Disparate Impact (≥ 0.80) | Demographic Parity Δ (< 0.10) | Equalized Odds Δ (< 0.10) | Max ECE (< 0.05) | Veredicto |
|-----------------|:------------------------:|:-----------------------------:|:-------------------------:|:----------------:|:---------:|
| `gender`        | **0.998** ✓              | **0.001** ✓                   | **0.012** ✓               | 0.156 ✗          | ⚠ solo ECE |
| `SeniorCitizen` | 0.519 ✗                  | 0.293 ✗                       | 0.245 ✗                   | 0.172 ✗          | ⚠ revisar |
| `Partner`       | 0.480 ✗                  | 0.250 ✗                       | 0.195 ✗                   | 0.173 ✗          | ⚠ revisar |
| `Dependents`    | 0.387 ✗                  | 0.276 ✗                       | 0.224 ✗                   | 0.175 ✗          | ⚠ revisar |

### 7.3 Interpretación

1. **`gender` — paridad casi perfecta** en selección, TPR y FPR. El atributo protegido más crítico del dataset no muestra disparidad de trato.
2. **Las disparidades de `SeniorCitizen`/`Partner`/`Dependents` reflejan diferencias reales de prevalencia**, no trato diferencial injustificado: los seniors churnean 44.4 % vs 23.1 %, y el modelo — correctamente — los selecciona más (61.0 % vs 31.6 %). Por el resultado de imposibilidad de Kleinberg et al. (2016), un modelo informativo **no puede** satisfacer simultáneamente paridad demográfica y calibración cuando los base rates difieren entre grupos.
3. **La naturaleza de la intervención importa**: la salida del modelo dispara **acciones beneficiosas** (ofertas de retención), no adversas. Seleccionar más a los grupos de mayor riesgo significa dirigirles **más** beneficios — el criterio de paridad demográfica no es el adecuado para este caso de uso, como se documenta en la decisión de mitigación de [`ethics_and_fairness.md`](ethics_and_fairness.md) §4.
4. **El ECE alto es global, no diferencial**: el ECE de test completo es 0.153 y los subgrupos están en 0.105-0.175 — consecuencia de `class_weight=balanced`, no de miscalibración selectiva contra un grupo. La brecha TPR de `Partner`/`Dependents` (~0.20) sí se monitorea como riesgo residual: los churners con pareja/dependientes tienen menor probabilidad de ser detectados (recall 0.55-0.60 vs 0.77-0.79).

---

## 8. Consideraciones éticas

Ver documento completo en [`ethics_and_fairness.md`](ethics_and_fairness.md). Resumen:

- El modelo se usa **exclusivamente** para priorizar acciones positivas.
- No se utiliza para denegación de servicio.
- Análisis de equidad ejecutado y publicado en esta model card (§7) **antes del release** `1.0.0`.
- Mecanismo de _override_ humano disponible.

---

## 9. Recomendaciones y advertencias

- **No leer las probabilidades crudas como frecuencias**: aplicar recalibración isotónica (ajustada sobre `val`) antes de cualquier uso que dependa de la probabilidad absoluta (p. ej. cálculo de valor esperado de campañas). La decisión binaria a threshold 0.58 no se ve afectada.
- **Monitorear la brecha de recall** de `Partner`/`Dependents` en producción (plan en [`ethics_and_fairness.md`](ethics_and_fairness.md) §6); si la brecha crece, evaluar thresholds por subgrupo o reponderación.
- El dataset es relativamente pequeño y antiguo; las predicciones pueden no generalizar a poblaciones contemporáneas sin un _retraining_ con datos representativos.
- El modelo no incluye señales de comportamiento (uso del producto, tickets, NPS); en producción real, estas señales serían críticas y deberían sumarse al pipeline.
- El umbral operativo debe revisarse por análisis de costo-beneficio del equipo de Retención; 0.58 maximiza F1, no necesariamente el ROI de cada campaña.

---

## 10. Citación

```bibtex
@misc{churnlens2026,
  author       = {Jhon Gallego},
  title        = {ChurnLens: Predicción temprana de churn en servicios por suscripción},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/jhonevergallegoate/churnlens}}
}
```

---

<sub>_Plantilla basada en: Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*'19. Resultado de imposibilidad: Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). Inherent Trade-Offs in the Fair Determination of Risk Scores._</sub>
