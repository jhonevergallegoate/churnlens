# Informe de salida — ChurnLens

> **Entregable de aceptación (Fase 5 · final)** del Proyecto Aplicado del Módulo 6 — Diplomado MLDS (UNAL). Estructura basada en el template TDSP del Mindlab (`docs/acceptance/exit_report.md`). Cierra el ciclo de las cinco fases del proyecto: entendimiento del negocio, preprocesamiento + EDA, modelamiento, despliegue y evaluación final.

---

## Resumen Ejecutivo

**ChurnLens** es un sistema de _machine learning_ para la **predicción temprana de churn** en servicios por suscripción, construido end-to-end sobre el dataset público _Telco Customer Churn_ (IBM, 7 043 clientes) siguiendo la metodología **TDSP**.

El resultado final es un servicio de inferencia **desplegado y reproducible**:

- **Modelo:** regresión logística L1 (`logreg_l1`), seleccionada por parsimonia entre 8 candidatos (incluyendo Random Forest, HistGradientBoosting y LightGBM, que **no la superaron**).
- **Desempeño held-out (test, nunca tocado hasta la Fase 4):** PR-AUC **0.6313** (2.4× el baseline trivial), ROC-AUC **0.8460**, recall **0.73**, **lift@10 % de 2.78×** — contactar el decil de mayor riesgo encuentra casi el triple de churners reales que contactar al azar.
- **Despliegue:** API REST (FastAPI) con contratos estrictos, empaquetada en imagen Docker multi-stage que **reconstruye el pipeline completo desde cero** (semilla 42) y produce probabilidades byte-equivalentes.
- **Gobernanza:** auditoría cuantitativa de _fairness_ sobre 4 atributos sensibles, model card completa, y CI de 9 jobs que valida desde el lint hasta el contenedor en HTTP real (142 tests, cobertura 85 %).

El proyecto se entrega **completo en sus cinco fases**, con cada fase reproducible vía `make phase2 … phase5` y trazable en la [bitácora del proyecto](../project_management/phase_log.md).

---

## Resultados del proyecto

### Entregables por etapa

| Fase | Entregables principales | Evidencia |
|------|-------------------------|-----------|
| **1 · Negocio + datos** | Project charter (18 secciones, objetivos SMART), diccionario de las 21 variables, código de carga validado (Pandera + MD5) | [`project_charter.md`](../business_understanding/project_charter.md) · [`data_dictionary.md`](../data/data_dictionary.md) · [`scripts/data_acquisition/`](../../scripts/data_acquisition/main.py) |
| **2 · Preprocesamiento + EDA** | 7 features derivadas, `ColumnTransformer` sin leakage, split estratificado 70/15/15, 9 figuras + 4 tablas de EDA, reporte de resumen | [`data_summary_report.md`](../data/data_summary_report.md) · [`src/churnlens/features/`](../../src/churnlens/features/) |
| **3 · Modelamiento** | Selección de features por consenso de 4 técnicas, catálogo de 8 modelos con CV 5-fold, threshold tuning, registro auditado con hashes | [`baseline_models.md`](../modeling/baseline_models.md) · [`final_model_report.md`](../modeling/final_model_report.md) |
| **4 · Despliegue** | API REST de 4 endpoints, Docker multi-stage reproducible, smoke E2E, documentación de infraestructura con costos | [`deploymentdoc.md`](../deployment/deploymentdoc.md) · [`infrastructure.md`](../deployment/infrastructure.md) |
| **5 · Evaluación final** | Auditoría de fairness (módulo + script + CI), model card completa con métricas held-out, este informe de salida | [`model_card.md`](../governance/model_card.md) · [`ethics_and_fairness.md`](../governance/ethics_and_fairness.md) · [`scripts/evaluation/fairness_audit.py`](../../scripts/evaluation/fairness_audit.py) |

### Evaluación del modelo final vs el modelo base

| Modelo | PR-AUC CV (k=5) | PR-AUC val | PR-AUC test | Lift@10 % test |
|--------|:---------------:|:----------:|:-----------:|:--------------:|
| `dummy_prior` (base trivial) | 0.265 | 0.266 | — | 1.0× |
| `random_forest` | 0.656 | 0.613 | — | — |
| `lightgbm` | 0.640 | 0.611 | — | — |
| **`logreg_l1`** (final) ★ | **0.661 ± 0.017** | **0.629** | **0.6313** | **2.78×** |

Tres hallazgos estructuran el resultado:

1. **El modelo final mejora 2.4× al baseline trivial** en la métrica primaria (PR-AUC 0.6313 vs 0.2649) y sostiene su desempeño en el held-out (val 0.6293 → test 0.6313, **sin degradación**).
2. **Ningún modelo no lineal superó a la regresión logística** — con ~7 K filas y variables estructurales, la señal disponible es esencialmente lineal. La L1 además apaga 8 de 35 features automáticamente (entre ellas `Partner`, con coeficiente exactamente 0), entregando interpretabilidad gratis.
3. **El threshold sintonizado en val (0.58) se sostiene en test** — el protocolo de no tocar `test` hasta la Fase 4 evitó el sobreajuste de threshold y validó la disciplina del split.

### Cumplimiento de los criterios de éxito (definidos en Fase 1)

| Criterio | Objetivo | Resultado test | Cumple |
|----------|:--------:|:--------------:|:------:|
| ROC-AUC | ≥ 0.85 | 0.8460 | ✗ marginal (−0.4 pp) |
| PR-AUC | ≥ 0.65 | 0.6313 | ✗ (−1.9 pp; CV sí: 0.661) |
| F1 (Yes) | ≥ 0.62 | 0.6145 | ✗ marginal (−0.6 pp) |
| Recall (Yes) | ≥ 0.70 | 0.7286 | ✓ |
| ECE | ≤ 0.05 | 0.1527 | ✗ |
| Lift top decil | ≥ 3.0× | 2.78× | ✗ marginal |

**Lectura honesta:** los umbrales de la Fase 1 resultaron ambiciosos para un dataset de este tamaño y naturaleza. Las métricas de _ranking_ quedaron a décimas del objetivo y el único incumplimiento material es la **calibración** (consecuencia conocida de `class_weight=balanced`: probabilidad media predicha 0.42 vs tasa real 0.26). La decisión binaria a threshold 0.58 no se ve afectada; la recalibración isotónica queda documentada como primer ítem de la v1.1.

### Auditoría de fairness

Sobre los 4 atributos sensibles comprometidos desde la Fase 1 ([resultados completos](../governance/ethics_and_fairness.md#31-resultados-de-la-auditoría-fase-5--test--threshold-058)):

- **`gender`: paridad casi perfecta** — Disparate Impact 0.998, brecha de equalized odds 0.012.
- Las disparidades de selección en `SeniorCitizen`/`Partner`/`Dependents` **reflejan diferencias reales de prevalencia** (los seniors churnean 44.4 % vs 23.1 %): el modelo dirige más ofertas de retención — una acción beneficiosa — a los grupos que más churnean. La decisión de **no** reponderar está justificada y documentada como desviación explícita en [`ethics_and_fairness.md` §4](../governance/ethics_and_fairness.md).
- **Riesgo residual bajo monitoreo:** brecha de recall (~0.20) para churners con pareja/dependientes.

### Relevancia para el negocio

Con la tasa base del 26.5 %, una campaña de retención no segmentada desperdicia ~3 de cada 4 contactos. Priorizando con ChurnLens, el decil superior concentra **73.6 % de churners reales** (lift 2.78×), y los drivers del modelo (contrato month-to-month, tenure corto, electronic check, sin servicios de soporte) son **directamente accionables** por Producto: migración a contratos anuales, onboarding temprano y promoción de pagos automáticos.

---

## Lecciones aprendidas

### Manejo de los datos

- **Validar en la frontera, no en el consumo:** el esquema Pandera + checksums MD5 en la adquisición eliminó toda una clase de errores aguas abajo. Los 11 NaN de `TotalCharges` (todos con `tenure = 0`) se detectaron en Fase 1, no en producción.
- **La imputación pertenece al `ColumnTransformer`**, no al dataframe: imputar la mediana *dentro* del pipeline (ajustada solo sobre `train`) fue lo que mantuvo el protocolo libre de leakage cuando los splits se rehacían.
- **El contrato del pipeline necesita auditoría propia:** `SeniorCitizen` quedó silenciosamente excluido del `ColumnTransformer` (no estaba en ningún bloque y `remainder="drop"` lo descartó). Se descubrió en la auditoría de fairness de la Fase 5 — un test que verifique "columnas declaradas = columnas consumidas" lo habría detectado en Fase 2.

### Modelamiento

- **La parsimonia gana cuando la señal es lineal:** invertir en baselines fuertes antes que en modelos complejos ahorró semanas — RF/LightGBM nunca superaron a la logística y el empate se detectó temprano gracias a la CV estratificada con desviaciones estándar.
- **`class_weight=balanced` cobra su precio en calibración:** resuelve el desbalance para el ranking pero infla las probabilidades. Si las probabilidades crudas importan (valor esperado de campañas), hay que recalibrar — lección incorporada a la model card como advertencia explícita.
- **Reservar `test` de verdad** (ni para threshold tuning) dio el resultado más valioso del proyecto: evidencia limpia de generalización.
- **Detalle operativo:** anidar `n_jobs=-1` dentro de `cross_val_score(n_jobs=-1)` crea n² workers y _live-lock_ con LightGBM. Paralelizar en un solo nivel.

### Implementación y despliegue

- **La imagen que reconstruye sus artefactos es su propia prueba de reproducibilidad:** el `docker build` reentrena desde cero y produce probabilidades byte-equivalentes — el rollback es un tag.
- **CI por fases** (un smoke job por fase TDSP, cada uno consumiendo los artefactos reales) detecta regresiones de integración que los tests unitarios no ven.
- **Los contratos estrictos en la API** (dominios cerrados + reglas de integridad replicadas del esquema Pandera) convierten los errores de datos del consumidor en `422` inmediatos en lugar de predicciones silenciosamente corruptas.

### Recomendaciones para futuros proyectos

1. Definir los umbrales de éxito con una **corrida exploratoria previa** — los nuestros se fijaron a ciegas en Fase 1 y resultaron ~2 pp por encima de lo alcanzable.
2. Presupuestar la **auditoría de fairness desde el diseño del pipeline** (qué columnas sensibles deben sobrevivir hasta la evaluación), no como paso final.
3. Incluir **señales de comportamiento** (uso, tickets, NPS) — las variables estructurales tocan techo alrededor de PR-AUC ~0.65 en este dominio.

---

## Impacto del proyecto

### Impacto en el negocio (simulado sobre el caso de uso)

- **Eficiencia de retención:** priorización que casi triplica la efectividad del contacto (lift 2.78×) y un threshold operable por variable de entorno para ajustar el volumen de campaña sin redeploy.
- **Drivers accionables:** la interpretabilidad de la L1 entrega palancas concretas a Producto (tipo de contrato, método de pago, onboarding temprano) además de la lista priorizada.
- **Infraestructura de decisión:** la API + model card + auditoría de fairness forman un paquete listo para revisión de un comité de gobernanza interno.

### Impacto académico

- Template TDSP del Mindlab **extendido** con artefactos de gobernanza (model card, ética/fairness, privacidad), CI multi-fase y registro de modelos auditado — reutilizable como referencia para próximas cohortes.
- Demostración práctica del **resultado de imposibilidad de Kleinberg et al.**: la tensión real entre paridad demográfica y calibración cuando los base rates difieren, resuelta con una decisión documentada en lugar de una métrica ciega.

### Áreas de mejora y desarrollo futuro

| Oportunidad | Acción propuesta |
|-------------|------------------|
| Calibración (ECE 0.15) | Recalibración isotónica ajustada sobre `val` (v1.1) |
| Brecha de recall en `Partner`/`Dependents` | Monitoreo en producción; evaluar thresholds por subgrupo si crece |
| Señal limitada del dataset | Incorporar telemetría de uso y señales de soporte |
| Despliegue real | Cloud Run con autenticación IAM (~$0-5/mes, plan en [`infrastructure.md`](../deployment/infrastructure.md)) |
| Medición causal | Diseño de experimento A/B para medir _uplift_ real de las campañas priorizadas |

---

## Conclusiones

1. El proyecto entregó un sistema de predicción de churn **completo, desplegado y auditado**, que cumple los entregables de las cinco rúbricas del Módulo 6 con reproducibilidad end-to-end (`make phase2 … phase5`, CI 9/9 verde, 142 tests, cobertura 85 %).
2. El modelo final supera 2.4× al baseline en la métrica primaria, generaliza sin degradación al held-out y produce un lift accionable de 2.78× — la **hipótesis central de negocio queda validada**: es posible priorizar la retención con variables estructurales del cliente.
3. Donde los números no alcanzaron los objetivos (calibración, décimas de PR-AUC/F1), el proyecto lo **reporta sin maquillaje** y deja el plan de cierre documentado — el criterio rector fue que un modelo honesto con limitaciones conocidas vale más que métricas infladas.
4. La metodología TDSP demostró su valor como contrato de comunicación: cada evaluador (y cada futuro mantenedor) encuentra los artefactos exactamente donde el proceso dice que deben estar.

---

## Agradecimientos

- Al **Mindlab y al equipo docente del Diplomado MLDS (Universidad Nacional de Colombia)** por el marco metodológico, el template TDSP y la retroalimentación durante las cinco fases.
- A **IBM** por la publicación del dataset _Telco Customer Churn_ que hizo posible un caso de uso realista sin comprometer datos personales.
- A la comunidad **open source** detrás de scikit-learn, pandas, FastAPI, Pandera y el ecosistema PyData, sobre la cual está construido cada componente de este proyecto.

---

<sub>_ChurnLens v1.0.0 · Jhon Gallego · Diplomado MLDS — UNAL · 2026._</sub>
