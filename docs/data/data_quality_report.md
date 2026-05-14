# Reporte preliminar de calidad de datos — ChurnLens

> **Propósito:** documentar el primer contacto numérico con el dataset, las anomalías detectadas durante la carga y validación, y las decisiones pendientes para la Fase 2.

> **Fecha de ejecución:** 2026-05-14
> **Dataset evaluado:** `data/raw/telco_customer_churn.csv`
> **MD5:** `3b0bfab28a8101b4e4fdd08025a5c235`
> **SHA-256:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
> **Tamaño en disco:** 970 457 bytes (~ 948 KB)

---

## 1. Resumen ejecutivo

| Métrica                                    | Valor                          |
|--------------------------------------------|--------------------------------|
| Filas totales                              | **7 043**                      |
| Columnas totales                           | **21**                         |
| Filas duplicadas                           | **0**                          |
| Columnas con valores constantes            | **0**                          |
| Valores faltantes (totales)                | **11**                         |
| Columnas con faltantes                     | **1** (`TotalCharges`)         |
| Tasa de la clase positiva (`Churn = Yes`)  | **26.54 %**                    |
| Razón de desbalance (negativos : positivos)| **2.77 : 1**                   |
| Validación de esquema (Pandera)            | ✅ **PASS**                    |
| Reglas de integridad cruzada               | ✅ **PASS**                    |

---

## 2. Validación contra el esquema Pandera

El esquema declarado en `src/churnlens/data/schema.py` impone:

- Tipos de dato exactos para cada columna (post-casteo).
- Dominios cerrados para todas las categóricas.
- Reglas de integridad cruzadas entre columnas dependientes.
- Unicidad del `customerID`.
- Formato regex del `customerID`.

Resultado de la corrida sobre el dataset completo:

```
$ churnlens data validate
[info] validating_schema    n_cols=21 n_rows=7043
[info] schema_ok            n_rows=7043
✓ Esquema válido — 7,043 filas × 21 columnas.
```

---

## 3. Análisis de valores faltantes

| Columna         | NaN | % del total | Causa raíz observada                                                 | Acción Fase 1                |
|-----------------|-----|-------------|----------------------------------------------------------------------|-------------------------------|
| `TotalCharges`  | 11  | 0.16 %      | Filas con `tenure = 0` (clientes recién dados de alta).              | Convertir `" "` → `NaN`.       |
| _Resto_         | 0   | 0.00 %      | —                                                                    | —                             |

> **Decisión:** los 11 `NaN` no se imputan en la Fase 1. Son **informativos** (indican clientes nuevos) y la imputación corresponde a la Fase 2, donde se decidirá si:
> - Reemplazar por 0 (justificado: si no han facturado nada, su `TotalCharges` real es 0).
> - Reemplazar por `MonthlyCharges × tenure`.
> - Crear una _flag_ binaria `was_total_charges_missing` y luego imputar.

---

## 4. Distribuciones univariadas (señalización temprana)

| Variable                | Resumen rápido                                                                                  |
|-------------------------|-------------------------------------------------------------------------------------------------|
| `tenure`                | Bimodal: gran masa en `[0, 6]` y otra cresta en `[60, 72]`. Muy informativo.                    |
| `MonthlyCharges`        | Aproximadamente bimodal: pico en USD 20 (clientes solo con teléfono) y plateau en USD 70–110.   |
| `TotalCharges`          | Distribución larga a la derecha; correlaciona con `tenure × MonthlyCharges`.                    |
| `Contract`              | 55 % `Month-to-month`, 21 % `One year`, 24 % `Two year`.                                        |
| `InternetService`       | 44 % `Fiber optic`, 34 % `DSL`, 22 % `No`.                                                       |
| `PaymentMethod`         | El más común es `Electronic check` (~34 %), seguido de los métodos automáticos.                  |
| `gender`                | Balanceado (50 / 50).                                                                            |
| `SeniorCitizen`         | 16 % senior.                                                                                     |
| `Churn` _(target)_      | **26.54 %** `Yes` (clase 1).                                                                     |

> **Observación clave:** la combinación `Contract = Month-to-month` + `InternetService = Fiber optic` + `tenure < 6` concentra a los clientes con mayor probabilidad histórica de churn. Esta señal se cuantificará rigurosamente en el EDA de la Fase 2.

---

## 5. Riesgos de integridad detectados

| Riesgo                                                                                                        | Detección | Decisión                                                              |
|---------------------------------------------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------|
| `MultipleLines = "No phone service"` cuando `PhoneService = Yes` (inconsistencia).                            | Validado  | Pandera lo rechaza con `phone_lines_consistency`.                      |
| Add-ons de internet con valores diferentes de `"No internet service"` cuando `InternetService = No`.          | Validado  | Pandera lo rechaza con `internet_addons_consistency`.                  |
| `customerID` duplicado.                                                                                       | Validado  | Pandera rechaza por `unique=True`.                                     |
| `TotalCharges` numérico negativo.                                                                              | Validado  | Pandera rechaza por `greater_than_or_equal_to(0)`.                     |
| `tenure` fuera del rango `[0, 72]`.                                                                            | Validado  | Pandera rechaza por `in_range(0, 72)`.                                 |
| Posible _data leakage_ entre `TotalCharges` y `Churn` (no chequeado aún).                                     | Pendiente | A revisar en EDA Fase 2.                                               |

---

## 6. Limitaciones conocidas del dataset

1. **Tamaño moderado** (7 043 filas) — limita la capacidad de generalización.
2. **Snapshot estático** — no permite analizar dinámica temporal real (no hay timestamps individuales por cliente).
3. **No incluye señales de comportamiento** (uso del producto, tickets de soporte, NPS, sesiones). Estas serían las _features_ más predictivas en un escenario productivo, pero están ausentes aquí.
4. **Sesgo de selección desconocido** — IBM no documenta cómo se construyó la muestra original.
5. **Antigüedad del dataset** — publicado originalmente alrededor de 2018; las distribuciones pueden no reflejar el mercado actual.

> Todas estas limitaciones se reportan también en la _model card_ (Fase 3+).

---

## 7. Decisiones de calidad para Fase 2

| Decisión                                                            | Estado     | Próximo paso                            |
|---------------------------------------------------------------------|------------|------------------------------------------|
| Tratamiento de `NaN` en `TotalCharges`                              | Pendiente  | Comparar 3 estrategias en EDA.            |
| Codificación de categóricas (OHE vs. ordinal vs. target encoding)   | Pendiente  | Benchmark Fase 2.                         |
| Manejo del desbalance (resampling / class_weight / threshold tuning)| Pendiente  | Probar las 3 estrategias en Fase 3.       |
| _Feature engineering_ derivado (`tenure_bucket`, `services_count`)  | Pendiente  | Diseño + ablation en Fase 2.              |
| Análisis de outliers en `MonthlyCharges` y `TotalCharges`           | Pendiente  | Boxplots + IQR-rule en EDA Fase 2.        |

---

## 8. Reproducibilidad

Para regenerar este reporte:

```bash
make data           # descarga + valida + materializa
churnlens data summary
churnlens data hash
```

El hash registrado en `data/raw/.checksums.json` permite confirmar que el archivo crudo evaluado en este reporte coincide bit a bit con el que cualquier evaluador descargue.

---

<sub>_Reporte preliminar — Diplomado MLDS, Universidad Nacional de Colombia._</sub>
