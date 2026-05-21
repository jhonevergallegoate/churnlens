# Reporte de resumen de los datos — ChurnLens

> **Fase 2 (10 %) — Preprocesamiento y Análisis Exploratorio.**
> Diplomado MLDS · Universidad Nacional de Colombia · Módulo 6.
> Fecha de generación: **2026-05-20**.
> Dataset: _Telco Customer Churn_ — IBM (7 043 filas × 21 columnas).
> Reproducible mediante `make phase2` o `churnlens eda report`.

---

## 1. Resumen ejecutivo

El dataset cubre **7 043 clientes únicos** con **21 variables** observadas
al momento del corte (1 identificador + 19 _features_ + 1 _target_). La
tasa global de cancelación es **26.54 %**, lo que configura un problema
de clasificación binaria **moderadamente desbalanceado** (≈ 2.77 :1 a
favor de la clase negativa).

El _target_ `Churn` se asocia con intensidad **moderada** a `Contract`,
`tenure`, los add-ons de internet, `InternetService` y `PaymentMethod`,
y prácticamente **no se asocia** con `gender` ni con `PhoneService`. La
única columna con valores faltantes es `TotalCharges` (11 filas,
0.16 %), todas correspondientes a clientes con `tenure = 0`.

Estos hallazgos se materializan, sin alterar los datos crudos, en un
**pipeline de preprocesamiento reproducible** que produce tres
particiones (`train` 70 % / `val` 15 % / `test` 15 %) preservando la
tasa de churn en cada conjunto (26.5 % ± 0.1 pp).

---

## 2. Procedencia y carga

- **Origen:** dataset _Telco Customer Churn_, publicado por IBM Cognos
  Analytics. Réplica usada: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Tamaño en disco:** 970 457 bytes (≈ 948 KiB).
- **Hash de integridad (MD5):** `3b0bfab28a8101b4e4fdd08025a5c235`.
- **Encoding:** UTF-8 sin BOM, separador `,`.
- **Carga oficial:** `churnlens.data.loader.TelcoChurnLoader.load_validated`
  ejecuta descarga → cast de tipos → validación Pandera → persistencia
  parquet.
- **Detalle por variable:** ver [`data_dictionary.md`](data_dictionary.md).

---

## 3. Estructura general

| Dimensión                              | Valor               |
|----------------------------------------|---------------------|
| Filas (clientes únicos)                | **7 043**           |
| Columnas totales                       | **21**              |
| Identificador                          | `customerID` (string sintético, único) |
| Variables numéricas continuas          | 3 (`tenure`, `MonthlyCharges`, `TotalCharges`) |
| Variables categóricas (incl. `SeniorCitizen` codificado como int8) | 17 |
| Variable objetivo                      | `Churn` (binaria, `Yes`/`No`) |
| Valores faltantes                      | 11 en `TotalCharges` (0.16 %), 0 en el resto |
| Filas duplicadas                       | 0                   |
| Reglas de integridad cruzada validadas | 2 (ver §3.1)        |

### 3.1 Reglas de integridad respetadas

- `PhoneService = No`  ⟹  `MultipleLines = "No phone service"` (1 526 filas).
- `InternetService = No`  ⟹  todos los _add-ons_ son `"No internet service"`
  (1 526 filas).

Ambas se validan en `RAW_SCHEMA` (`src/churnlens/data/schema.py`).

---

## 4. Variable objetivo

| Clase   | Conteo | Porcentaje |
|---------|--------|------------|
| `No`    | 5 174  | **73.46 %** |
| `Yes`   | 1 869  | **26.54 %** |
| **Total** | 7 043 | 100.00 %   |

![Distribución del target](../../reports/figures/eda_target_distribution.png)

**Implicación de modelado.** El desbalance es moderado: no se justifica
oversampling agresivo, pero sí preferir métricas robustas (PR-AUC, F1
sobre la clase positiva) y considerar `class_weight='balanced'` o
calibración del _threshold_ en la Fase 3.

---

## 5. Estadísticas descriptivas — variables numéricas

| Variable         | n     | NaN | mean   | std    | min   | p05   | p25   | p50   | p75   | p95    | max     | skew | kurtosis |
|------------------|-------|-----|--------|--------|-------|-------|-------|-------|-------|--------|---------|------|----------|
| `tenure` (meses) | 7 043 | 0   | 32.37  | 24.56  | 0     | 1     | 9     | 29    | 55    | 72     | 72      | 0.24 | -1.39    |
| `MonthlyCharges` | 7 043 | 0   | 64.76  | 30.09  | 18.25 | 19.65 | 35.50 | 70.35 | 89.85 | 107.40 | 118.75  | -0.22| -1.26    |
| `TotalCharges`   | 7 032 | 11  | 2 283.30 | 2 266.77 | 18.80 | 49.61 | 401.45 | 1 397.48 | 3 794.74 | 6 923.59 | 8 684.80 | 0.96 | -0.23 |

> Fuente: `reports/tables/eda_numeric_summary.csv` (generado por `churnlens eda report`).

### 5.1 Lecturas clave

- **`tenure`** es bimodal con cúmulos en los extremos (`0-12m` y `49-72m`,
  ver §6.3) y curtosis fuertemente negativa (-1.39) — comportamiento
  consistente con una mezcla de clientes nuevos y clientes consolidados.
- **`MonthlyCharges`** está casi uniforme entre 20 y 110 USD con leve
  sesgo a la izquierda; representa la heterogeneidad de planes
  contratados (con / sin internet, con / sin add-ons).
- **`TotalCharges`** es altamente sesgada a la derecha (skew = 0.96)
  porque acumula el gasto histórico — los clientes con baja antigüedad
  dominan los percentiles bajos.
- La correlación Spearman entre `tenure` y `TotalCharges` es **0.89**
  (alta, esperada): `TotalCharges ≈ MonthlyCharges × tenure`.

![Distribución de tenure](../../reports/figures/eda_tenure_histogram.png)
![Distribución de MonthlyCharges](../../reports/figures/eda_monthly_charges_histogram.png)
![TotalCharges por target](../../reports/figures/eda_total_charges_box.png)

---

## 6. Bivariado — tasa de churn por variable

Se reportan las tres variables categóricas con mayor asociación al
_target_, medida con la **V de Cramér** corregida (Bergsma-Wicher),
listada de mayor a menor.

| Variable categórica | V de Cramér | Comentario |
|---------------------|-------------|------------|
| `Contract`          | **0.410**   | Predictor más fuerte. |
| `tenure_bucket`     | **0.348**   | Bucket derivado en preprocesamiento. |
| `OnlineSecurity`    | 0.347       | El "no tiene" es el modo y churnea más. |
| `TechSupport`       | 0.343       | Comportamiento simétrico al anterior. |
| `InternetService`   | 0.322       | _Fiber optic_ churnea 5.6× más que "sin internet". |
| `PaymentMethod`     | 0.303       | _Electronic check_ es el método más volátil. |
| `OnlineBackup`      | 0.292       |                                    |
| `DeviceProtection`  | 0.281       |                                    |
| `StreamingMovies`   | 0.230       |                                    |
| `StreamingTV`       | 0.230       |                                    |
| `PaperlessBilling`  | 0.191       |                                    |
| `Dependents`        | 0.164       |                                    |
| `Partner`           | 0.150       |                                    |
| `MultipleLines`     | 0.036       | Asociación marginal.               |
| `PhoneService`      | 0.001       | Práctica­mente nula.               |
| `gender`            | 0.000       | Nula — **descartar como feature**. |

> Fuente: `reports/tables/eda_cramers_v.csv`.

### 6.1 Churn por modalidad de contrato

| Contract        | Clientes | Churn rate |
|-----------------|----------|------------|
| Month-to-month  | 3 875    | **42.71 %** |
| One year        | 1 473    | 11.27 %    |
| Two year        | 1 695    | **2.83 %** |

Diferencia de **39.9 puntos porcentuales** entre los extremos — el
contrato es el factor de _lock-in_ más potente del dataset.

![Churn por Contract](../../reports/figures/eda_churn_by_contract.png)

### 6.2 Churn por método de pago

| PaymentMethod              | Clientes | Churn rate |
|----------------------------|----------|------------|
| Electronic check           | 2 365    | **45.29 %** |
| Mailed check               | 1 612    | 19.11 %    |
| Bank transfer (automatic)  | 1 544    | 16.71 %    |
| Credit card (automatic)    | 1 522    | 15.24 %    |

El _Electronic check_ es el único método **manual y digital** y duplica
la tasa de los métodos automatizados — motivación clara para la
_feature_ derivada `auto_payment`.

![Churn por PaymentMethod](../../reports/figures/eda_churn_by_payment_method.png)

### 6.3 Churn por bucket de antigüedad (feature derivada)

| `tenure_bucket` | Clientes | Churn rate |
|-----------------|----------|------------|
| 0-12m           | 2 186    | **47.44 %** |
| 13-24m          | 1 024    | 28.71 %    |
| 25-48m          | 1 594    | 20.39 %    |
| 49-72m          | 2 239    | 9.51 %     |

La probabilidad de churn cae **5×** entre clientes nuevos (≤ 1 año) y
clientes consolidados (4-6 años). La feature `tenure_bucket` resume
esta no-linealidad de forma legible para modelos lineales.

![Churn por tenure_bucket](../../reports/figures/eda_churn_by_tenure_bucket.png)

---

## 7. Correlaciones entre variables numéricas

| Par                                   | Spearman | Interpretación |
|---------------------------------------|----------|----------------|
| `tenure` × `TotalCharges`             | **0.89** | Esperada (acumulación). |
| `MonthlyCharges` × `avg_monthly_spend`| **0.99** | Casi colineales — `avg_monthly_spend` solo aporta cuando hay cambios de plan. |
| `MonthlyCharges` × `services_count`   | 0.82     | A más servicios, más cargo. |
| `services_count` × `TotalCharges`     | 0.77     | Idem agregado en el tiempo. |
| `tenure` × `MonthlyCharges`           | 0.28     | Baja — el cargo mensual no escala con la antigüedad. |
| `monthly_spend_gap` × resto           | ≈ 0      | Diseñada para ser **ortogonal**: detecta upsell / downsell aislados. |

![Matriz de correlación Spearman](../../reports/figures/eda_correlation_heatmap.png)

> **Decisión.** Se conservan ambas (`MonthlyCharges` y
> `avg_monthly_spend`) porque la redundancia no penaliza modelos basados
> en árboles y `avg_monthly_spend` permite separar cambios de plan en
> modelos lineales. Se evita usar simultáneamente sin regularización
> (Fase 3).

---

## 8. Valores faltantes

| Columna        | NaN | % del total |
|----------------|-----|-------------|
| `TotalCharges` | 11  | 0.16 %      |
| _resto_        | 0   | 0 %         |

Las 11 filas con `TotalCharges = NaN` cumplen todas que `tenure = 0`
(clientes recién dados de alta sin facturación acumulada). La política
elegida (ver §9) es **imputar con la mediana en `train`** vía
`SimpleImputer`, dentro del `ColumnTransformer`, para evitar _leakage_
y mantener el _step_ trazable.

![Valores faltantes](../../reports/figures/eda_missing_values.png)

---

## 9. Decisiones de preprocesamiento

| Decisión                                         | Justificación |
|--------------------------------------------------|---------------|
| **No imputar antes del split.** Imputar dentro del `ColumnTransformer` con `SimpleImputer(strategy="median")`, ajustado **solo** sobre `train`. | Evita leakage entre `train` y `val`/`test`. La mediana es robusta a la cola larga de `TotalCharges`. |
| **Escalar numéricas con `StandardScaler`.** | Necesario para modelos lineales y SVM en Fase 3; transparente para árboles. |
| **`Contract` y `tenure_bucket` como ordinales.** | Tienen orden natural (`Month-to-month < One year < Two year`; buckets crecientes). |
| **Binarias `Yes/No`, `gender`, booleanos derivados → `OrdinalEncoder`.** | Compactas, sin explosión dimensional, orden explícito y reproducible. |
| **Nominales multi-clase → `OneHotEncoder(drop="first")`.** | Evita dummy-trap en modelos lineales; tolerante a categorías desconocidas en inferencia (`handle_unknown="ignore"`). |
| **`gender` y `PhoneService` no se eliminan en Fase 2** pese a V de Cramér ≈ 0. | La selección de features se hace **dentro del experimento de modelado** (Fase 3) con criterios reproducibles (información mutua, permutation importance). |
| **Split 70/15/15 estratificado por `Churn`, `seed = 42`.** | Tasa de churn preservada en los tres conjuntos (26.5 % ± 0.1 pp). |

### 9.1 Features derivadas (módulo `churnlens.features.engineering`)

| Feature             | Tipo          | Definición                                                  | Hipótesis de negocio |
|---------------------|---------------|-------------------------------------------------------------|----------------------|
| `tenure_bucket`     | `category` ord.| `pd.cut(tenure, [-1, 12, 24, 48, 72])`                      | Resume la no-linealidad del riesgo de churn vs antigüedad. |
| `services_count`    | `int8` `[0,8]` | Conteo de servicios y add-ons activos.                      | _Proxy_ de lock-in: a más servicios contratados, mayor costo de cambio. |
| `has_internet`      | `bool`         | `InternetService != "No"`                                   | Bandera explícita para árboles. |
| `has_phone`         | `bool`         | `PhoneService == "Yes"`                                     | Idem. |
| `auto_payment`      | `bool`         | `PaymentMethod ∈ {Bank transfer, Credit card}`              | Captura la fricción mecánica del pago automático. |
| `avg_monthly_spend` | `float32`      | `TotalCharges / max(tenure, 1)`                             | Detecta cambios de plan a lo largo de la antigüedad. |
| `monthly_spend_gap` | `float32`      | `MonthlyCharges - avg_monthly_spend`                        | Captura _upsell_ (+) o _downsell_ (−) reciente. |

---

## 10. Artefactos producidos

| Ruta                                                    | Contenido |
|---------------------------------------------------------|-----------|
| `data/processed/train.parquet`                          | 4 929 filas × 36 cols (35 features + target). |
| `data/processed/val.parquet`                            | 1 057 filas × 36 cols. |
| `data/processed/test.parquet`                           | 1 057 filas × 36 cols. |
| `data/processed/preprocessor.joblib`                    | `ColumnTransformer` ajustado a `train`. |
| `data/processed/feature_names.json`                     | Nombres post-transformación. |
| `data/processed/metadata.json`                          | Shapes, tasa de positivos por split, semilla, fecha. |
| `reports/figures/eda_*.png`                             | 9 figuras del EDA. |
| `reports/tables/eda_*.csv`                              | 4 tablas (resumen numérico, categórico, V de Cramér, target). |

---

## 11. Reproducibilidad

Cualquier evaluador puede regenerar **bit-equivalentes** todos los
artefactos anteriores ejecutando:

```bash
# Instalar en modo editable con extras de notebooks
pip install -e ".[dev,notebooks]"

# Pipeline Fase 1 + Fase 2 completo
make phase2

# O paso a paso
churnlens data download
churnlens data validate
churnlens eda report
churnlens preprocess run
```

Garantías:

- Misma URL de descarga + hash MD5 verificado → mismos bytes crudos.
- Misma semilla (`CHURNLENS_RANDOM_SEED=42`) → misma partición.
- `ColumnTransformer` serializado en `preprocessor.joblib` → mismas
  transformaciones en inferencia futura.

---

## 12. Conclusiones para Fase 3

1. **Predictores fuertes confirmados:** `Contract`, `tenure`,
   `InternetService` y los add-ons de seguridad/soporte. La feature
   derivada `tenure_bucket` resume esta señal de forma legible.
2. **Predictores a descartar tentativamente:** `gender` (V Cramér = 0)
   y `PhoneService` (V Cramér ≈ 0). La decisión final se toma con
   selección de features cuantitativa en Fase 3.
3. **Métricas a usar:** PR-AUC y F1 sobre la clase positiva — la
   accuracy global no captura la utilidad práctica (un clasificador
   constante obtendría 73 %).
4. **Calibración del _threshold_** será necesaria si se reporta
   probabilidad esperada de churn al área de retención.
5. **No-imputación previa al split** está bloqueada por diseño: la
   imputación vive dentro del pipeline sklearn, garantizando que no
   haya información de `val` / `test` en el entrenamiento.

---

## 13. Referencias internas

- [`data_dictionary.md`](data_dictionary.md) — diccionario detallado por variable.
- [`data_definition.md`](data_definition.md) — origen, licencia, rutas y casteo.
- [`data_quality_report.md`](data_quality_report.md) — calidad preliminar (Fase 1).
- `src/churnlens/features/engineering.py` — features derivadas.
- `src/churnlens/features/preprocessing.py` — `ColumnTransformer`.
- `src/churnlens/features/pipeline.py` — orquestador end-to-end.
- `src/churnlens/eda/` — estadísticas, plots y orquestador.
- `notebooks/02_eda_and_preprocessing.ipynb` — narrativa exploratoria.

---

<sub>_Documento del entregable de Fase 2 — Diplomado MLDS, Universidad Nacional de Colombia._</sub>
