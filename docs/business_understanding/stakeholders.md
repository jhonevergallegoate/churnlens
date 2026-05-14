# Mapa de Stakeholders — ChurnLens

> **Propósito:** identificar a todos los actores con interés, influencia o impacto sobre el proyecto, segmentarlos por su nivel de poder/interés y definir la estrategia de comunicación con cada uno.

---

## 1. Inventario de stakeholders

| ID    | Stakeholder                                | Tipo        | Rol en el proyecto                                                            |
|-------|---------------------------------------------|-------------|-------------------------------------------------------------------------------|
| S-01  | Coordinador del Módulo 6                    | Académico   | Evalúa el cumplimiento de la rúbrica y aprueba cada fase.                     |
| S-02  | Equipo docente del Diplomado MLDS           | Académico   | Acompaña la metodología TDSP y resuelve dudas técnicas.                       |
| S-03  | Estudiante / autor del proyecto             | Interno     | Diseña, ejecuta, documenta y entrega.                                          |
| S-04  | Comunidad académica del Diplomado           | Externo     | Consume el repositorio como referencia de buenas prácticas.                    |
| S-05  | Equipo de Retención (rol simulado)          | Negocio     | Usuario final del modelo en un escenario productivo simulado.                  |
| S-06  | Equipo de Producto (rol simulado)           | Negocio     | Consume _insights_ sobre _drivers_ de _churn_ para decisiones de _packaging_. |
| S-07  | Equipo Financiero (rol simulado)            | Negocio     | Mide el impacto sobre _MRR_ retenido y ROI.                                   |
| S-08  | Oficial de privacidad / Legal (rol simulado)| Governance  | Verifica que el sistema cumpla normativas de protección de datos.              |
| S-09  | Comunidad de _open source_ / IBM            | Externo     | Proveedor original del dataset.                                                |

---

## 2. Matriz Poder / Interés

```
ALTO
poder │
      │     S-08            S-01
      │   (LEGAL)         (COORD)
      │
      │     S-07            S-05
      │  (FINANZAS)      (RETENCIÓN)
      │
      │                     S-03
      │                  (ESTUDIANTE)
      │
      │     S-09           S-02 · S-06
      │  (IBM/OSS)       (DOCENTES, PRODUCTO)
      │                     S-04
      │                  (COMUNIDAD)
BAJO  │
      └──────────────────────────────── ALTO
                       interés
```

- **Manage closely (alto poder + alto interés):** S-01 (coordinación), S-05 (retención), S-08 (legal).
- **Keep satisfied (alto poder + bajo interés):** S-07 (finanzas).
- **Keep informed (bajo poder + alto interés):** S-02 (docentes), S-06 (producto), S-03 (estudiante).
- **Monitor (bajo poder + bajo interés):** S-04 (comunidad), S-09 (IBM/OSS).

---

## 3. Expectativas y entregables por stakeholder

| Stakeholder | Expectativa principal                                              | Entregable que la satisface                                              |
|-------------|--------------------------------------------------------------------|--------------------------------------------------------------------------|
| S-01        | Cumplimiento de la rúbrica de la Fase 1 (10 %).                    | `project_charter.md`, diccionarios, código de carga funcional.            |
| S-02        | Aplicación correcta de TDSP.                                       | Estructura de carpetas alineada al template Mindlab.                      |
| S-03        | Proyecto de portafolio reproducible.                               | Repositorio público, README detallado, buenas prácticas demostradas.      |
| S-04        | Repositorio claro y aprendible.                                    | Documentación abundante, código tipado, tests, CI.                        |
| S-05        | Lista priorizada de clientes en riesgo, accionable.                | API REST + reporte tabular (Fase 4).                                      |
| S-06        | _Insights_ sobre por qué cancelan los clientes.                    | Análisis de _feature importance_ + _SHAP_ (Fase 3).                       |
| S-07        | Cuantificación del impacto económico.                              | `business_case.md` + monitoreo de _MRR_ retenido (Fase 4).                |
| S-08        | Garantía de no uso de PII y de fairness.                           | `privacy_and_compliance.md` · `ethics_and_fairness.md` · _model card_.    |
| S-09        | Atribución correcta del dataset.                                   | Licencia y atribución explícita en `data_definition.md` y `LICENSE`.      |

---

## 4. Plan de comunicación

| Stakeholder | Canal                       | Frecuencia                      | Formato                                  |
|-------------|-----------------------------|---------------------------------|------------------------------------------|
| S-01, S-02  | Repositorio GitHub          | Al cierre de cada fase TDSP     | _Pull Request_ + _tag_ semántico (`v0.1.0-fase1`). |
| S-03        | _Self_ — diario             | Diario                          | Commits + bitácora en `references/`.     |
| S-05–S-07   | Reporte ejecutivo simulado  | Cierre de Fase 3 y Fase 5       | PDF + presentación + demo.               |
| S-04        | README + Releases           | Cada entrega                    | Markdown público.                        |
| S-08        | Documentos de governance    | Cierre de Fase 3 y Fase 5       | _Model card_ + reporte de _fairness_.    |

---

## 5. RACI matriz (responsabilidades clave)

| Actividad                                | Estudiante (S-03) | Docentes (S-02) | Coordinación (S-01) |
|------------------------------------------|:-----------------:|:---------------:|:--------------------:|
| Definir alcance                          | **R/A**           | C               | I                    |
| Construir Project Charter                | **R/A**           | C               | I                    |
| Implementar carga de datos               | **R/A**           | I               | I                    |
| Construir diccionarios                   | **R/A**           | C               | I                    |
| Aprobar entrega Fase 1                   | I                 | C               | **A**                |
| Evaluar entrega Fase 1                   | I                 | C               | **R**                |

> **R**: _Responsible_ · **A**: _Accountable_ · **C**: _Consulted_ · **I**: _Informed_.
