# Model Card — ChurnLens

> **Estado en la Fase 1:** _esqueleto inicial._ Las secciones de métricas, _slices_ y consideraciones detalladas se diligencian al cierre de la **Fase 3** (modelado) y se actualizan en la **Fase 5** (entrega final). Estructura basada en _Mitchell et al., 2019 — Model Cards for Model Reporting_.

---

## 1. Identidad del modelo

| Campo                       | Valor                                                                                |
|-----------------------------|--------------------------------------------------------------------------------------|
| Nombre del modelo           | _ChurnLens-Baseline_ (TBD — definir nombre técnico final en Fase 3)                  |
| Versión                     | `0.0.0` (no entrenado aún)                                                            |
| Tipo de problema            | Clasificación binaria supervisada                                                     |
| Variable objetivo           | `Churn ∈ {Yes, No}`                                                                   |
| Dominio                     | Servicios por suscripción (telco / SaaS / fintech / streaming, etc.)                  |
| Fecha de _release_           | TBD                                                                                   |
| Autor / Propietario         | Jhon Gallego                                                                          |
| Licencia                    | MIT                                                                                   |
| Repositorio                 | `https://github.com/jhonevergallegoate/churnlens`                                      |

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

- `Contract`, `tenure`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, _add-ons_ del servicio.

### 3.2 Atributos sensibles a evaluar

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`.

---

## 4. Métricas

> **Pendiente Fase 3.** Esta sección se completa después del entrenamiento. Los umbrales objetivo se definen en [`success_criteria.md`](../business_understanding/success_criteria.md).

| Métrica                       | Train | CV (k=5) | Holdout | Umbral mínimo |
|-------------------------------|:-----:|:--------:|:-------:|:-------------:|
| ROC-AUC                       | TBD   | TBD      | TBD     | ≥ 0.85        |
| PR-AUC                        | TBD   | TBD      | TBD     | ≥ 0.65        |
| F1 (clase Yes)                | TBD   | TBD      | TBD     | ≥ 0.62        |
| Recall (clase Yes)            | TBD   | TBD      | TBD     | ≥ 0.70        |
| ECE (calibración)             | TBD   | TBD      | TBD     | ≤ 0.05        |
| Lift top decil                | TBD   | TBD      | TBD     | ≥ 3.0×        |

---

## 5. Datos de entrenamiento

| Campo                       | Valor                                                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------------------------------|
| Dataset                     | _Telco Customer Churn_ (IBM Sample Data Sets).                                                                  |
| Tamaño                      | 7 043 filas × 21 columnas.                                                                                      |
| Partición                   | TBD (Fase 3 — `train/val/test` estratificada por `Churn`).                                                       |
| Tasa de positivos           | 26.54 %.                                                                                                        |
| Documentación               | [`data_dictionary.md`](../data/data_dictionary.md) · [`data_definition.md`](../data/data_definition.md) · [`data_quality_report.md`](../data/data_quality_report.md). |

---

## 6. Datos de evaluación

> **Pendiente Fase 3.** Se reservará un _holdout_ estratificado del 20 % nunca tocado durante _tuning_.

---

## 7. Análisis cuantitativo de _fairness_

> **Pendiente Fase 3.** Se reportará para `gender`, `SeniorCitizen`, `Partner`, `Dependents` con las métricas descritas en [`ethics_and_fairness.md`](ethics_and_fairness.md).

| Subgrupo                | Tasa de selección | Disparate Impact | Equalized Odds Δ |
|-------------------------|:-----------------:|:----------------:|:----------------:|
| gender = Female         | TBD               | TBD              | TBD              |
| gender = Male           | TBD               | TBD              | TBD              |
| SeniorCitizen = 1       | TBD               | TBD              | TBD              |
| SeniorCitizen = 0       | TBD               | TBD              | TBD              |

---

## 8. Consideraciones éticas

Ver documento completo en [`ethics_and_fairness.md`](ethics_and_fairness.md). Resumen:

- El modelo se usa **exclusivamente** para priorizar acciones positivas.
- No se utiliza para denegación de servicio.
- Análisis de equidad obligatorio antes de cualquier "release" interno.
- Mecanismo de _override_ humano disponible.

---

## 9. Recomendaciones y advertencias

- El dataset es relativamente pequeño y antiguo; las predicciones pueden no generalizar a poblaciones contemporáneas sin un _retraining_ con datos representativos.
- El modelo no incluye señales de comportamiento (uso del producto, tickets, NPS); en producción real, estas señales serían críticas y deberían sumarse al pipeline.
- El umbral operativo debe definirse por análisis de costo-beneficio del equipo de Retención, no por _default_ del modelo.

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

<sub>_Plantilla basada en: Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*'19._</sub>
