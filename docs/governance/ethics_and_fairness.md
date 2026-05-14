# Ética y equidad algorítmica — ChurnLens

> **Propósito:** dejar por escrito los principios éticos del proyecto, identificar riesgos algorítmicos potenciales, y comprometerse con un plan explícito de evaluación de equidad sobre atributos sensibles.

---

## 1. Principios rectores

1. **No-discriminación accionable.** Las predicciones del modelo se usan exclusivamente para **priorizar** acciones positivas de retención. No se utilizan, bajo ninguna circunstancia, para denegar servicio, encarecer el precio individualmente o restringir capacidades del cliente.
2. **Transparencia.** Todas las decisiones de diseño se documentan; el código es trazable, los modelos son auditables, y las limitaciones se reportan en la _model card_.
3. **Privacidad por diseño.** Ninguna información personal identificable (PII) entra al modelo. El `customerID` del dataset es sintético.
4. **Minimización.** Se utiliza el conjunto mínimo de variables necesario para la tarea. Variables sensibles entran al análisis **solo para auditar sesgo**, no como _features_ predictivas directas.
5. **Reproducibilidad.** Cualquier resultado del modelo puede ser regenerado por terceros siguiendo el `README.md`.
6. **Reversibilidad.** Cualquier decisión automatizada apoyada por el modelo es reversible por un humano.

---

## 2. Atributos sensibles considerados

El dataset _Telco Customer Churn_ contiene los siguientes atributos que podrían considerarse sensibles bajo el _AI Act_ europeo, la ECOA (EEUU) o la Ley 1581 de 2012 (Colombia):

| Atributo          | Razón de sensibilidad                                                          | Estatus regulatorio típico |
|-------------------|--------------------------------------------------------------------------------|----------------------------|
| `gender`          | Atributo protegido directamente.                                               | Sensible (todas las jurisd.)|
| `SeniorCitizen`   | _Proxy_ de edad — atributo protegido directamente.                             | Sensible                    |
| `Partner`         | Estado civil aproximado; en algunas jurisdicciones es atributo protegido.       | Sensible (parcial)          |
| `Dependents`      | _Proxy_ de estado familiar.                                                    | Sensible (parcial)          |

---

## 3. Análisis comprometido

Para cada atributo sensible se realizará, a más tardar al cierre de la **Fase 3**, el siguiente análisis:

| Métrica                                 | Definición                                                                                                | Umbral objetivo        |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------|
| **Tasa de selección**                   | Fracción del grupo predicha como _churner_ por encima del umbral operativo.                                | Reportada              |
| **Disparate Impact (DI)**               | Razón entre tasa de selección del grupo desfavorecido y del grupo favorecido. Regla 80 % de la EEOC.       | DI ∈ **[0.80, 1.25]**  |
| **Equalized Odds difference**           | Máx. diferencia de TPR y FPR entre grupos.                                                                 | < **0.10**             |
| **Demographic Parity difference**       | Máx. diferencia de tasa de selección entre grupos.                                                         | < **0.10**             |
| **Calibración por subgrupo**            | ECE comparado por grupo.                                                                                   | < **0.05** en todos    |

Las métricas se reportan en la _model card_ (`docs/governance/model_card.md`) junto con la metodología utilizada para calcularlas y los datos de los subgrupos.

---

## 4. Decisiones explícitas de _fairness_

| Decisión                                                                              | Estatus                              |
|---------------------------------------------------------------------------------------|--------------------------------------|
| Incluir `gender` y `SeniorCitizen` como _features_ del modelo de baseline.            | **Sí**, para luego comparar con su exclusión y medir trade-offs de precisión vs equidad. |
| Aplicar _mitigation_ pre-procesamiento (reponderación) si DI < 0.80.                   | **Sí**, condicional al resultado de auditoría.                                            |
| Aplicar _mitigation_ post-procesamiento (calibración por subgrupo) si ECE diverge.    | **Sí**, condicional.                                                                       |
| Usar el modelo para automatizar denegación o sobrecargo individual.                    | **NO** — fuera del alcance permitido.                                                     |

---

## 5. Limitaciones conocidas

- El dataset es relativamente pequeño (~7 K filas), lo que limita la potencia estadística del análisis de _fairness_ sobre subgrupos pequeños (ej. `Dependents = Yes` × `SeniorCitizen = 1`).
- El dataset proviene de un proveedor con sesgo geográfico y de mercado desconocido. Las conclusiones de _fairness_ aplican a la distribución observada y no se extrapolan automáticamente a otros contextos.
- El `gender` del dataset es binario, lo cual no captura la realidad de género contemporánea. Esta limitación se documenta en la _model card_.

---

## 6. Plan de monitoreo en producción (Fase 4)

Una vez desplegado el servicio:

- **Drift de _features_ sensibles** se monitorea con PSI; alerta si PSI > 0.10 en alguna ventana semanal.
- **Métricas de _fairness_** se recalculan mensualmente con datos en vivo y se publican en el _dashboard_ interno.
- **Revisión humana** del top decil de cada ciclo, con muestreo aleatorio del 1 % para auditoría cualitativa.
- **Mecanismo de _override_** del modelo disponible para agentes humanos.

---

## 7. Referencias

1. Mitchell, M. et al. (2019). _Model Cards for Model Reporting_. FAT*'19.
2. Barocas, S., Hardt, M., & Narayanan, A. (2023). _Fairness and Machine Learning_.
3. EEOC (1978). _Uniform Guidelines on Employee Selection Procedures_ (80 % rule).
4. Unión Europea (2024). _AI Act — Regulation (EU) 2024/1689_.
5. República de Colombia (2012). _Ley 1581 de 2012 — Protección de Datos Personales_.
