# Diccionario de datos — ChurnLens

> **Dataset:** _Telco Customer Churn_ — IBM Sample Data Sets
> **Filas:** 7 043
> **Columnas:** 21 (20 _features_ + 1 _target_)
> **Granularidad:** un registro por cliente único (`customerID` es primary key sintético).
> **Idioma original de las variables:** Inglés.
> **Última actualización:** 2026-05-14.

---

## 1. Estructura del dataset

El dataset contiene **una sola tabla** llamada `customers` con 21 columnas. No hay relaciones con otras tablas dentro del alcance del proyecto.

### 1.1 Resumen de columnas

| Bloque                 | Columnas                                                                                                 |
|------------------------|----------------------------------------------------------------------------------------------------------|
| Identificador          | `customerID`                                                                                             |
| Demográficas           | `gender`, `SeniorCitizen`, `Partner`, `Dependents`                                                       |
| Cuenta / Contrato      | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`              |
| Servicio telefónico    | `PhoneService`, `MultipleLines`                                                                          |
| Servicio de internet   | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Variable objetivo      | `Churn` (target)                                                                                         |

### 1.2 Distribución de tipos

| Tipo lógico              | Conteo | Variables                                                                                                 |
|--------------------------|--------|-----------------------------------------------------------------------------------------------------------|
| Identificador (string)   | 1      | `customerID`                                                                                              |
| Numérica entera          | 2      | `SeniorCitizen`, `tenure`                                                                                 |
| Numérica continua        | 2      | `MonthlyCharges`, `TotalCharges`                                                                          |
| Categórica binaria       | 6      | `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`                            |
| Categórica multi-clase   | 10     | `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod` |

---

## 2. Diccionario detallado — Tabla `customers`

> **Nota sobre nulos:** El dataset crudo de IBM **no contiene valores `NaN`** explícitos. Sin embargo, la columna `TotalCharges` contiene **11 cadenas vacías (`" "`)** que deben ser tratadas como faltantes durante la carga.

| #  | Variable             | Tipo (cargado)      | Descripción                                                                                                              | Valores posibles / Rango                                                                                                   | Nulos esperados                                | Fuente |
|----|----------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|--------|
| 1  | `customerID`         | `string`            | Identificador único del cliente (sintético, no es PII).                                                                  | `^\d{4}-[A-Z]{5}$` (ej. `7590-VHVEG`)                                                                                       | 0                                              | IBM    |
| 2  | `gender`             | `category`          | Género auto-reportado del titular de la cuenta.                                                                          | `{Female, Male}`                                                                                                            | 0                                              | IBM    |
| 3  | `SeniorCitizen`      | `int8` (booleano)   | Indicador de si el cliente tiene 65+ años.                                                                                | `{0, 1}` _(1 = senior)_                                                                                                     | 0                                              | IBM    |
| 4  | `Partner`            | `category`          | Indica si el cliente tiene pareja (cónyuge / convivencia).                                                                | `{Yes, No}`                                                                                                                 | 0                                              | IBM    |
| 5  | `Dependents`         | `category`          | Indica si el cliente tiene dependientes (hijos, padres a cargo, etc.).                                                    | `{Yes, No}`                                                                                                                 | 0                                              | IBM    |
| 6  | `tenure`             | `int16`             | Antigüedad de la suscripción en **meses** desde el alta inicial.                                                          | `[0, 72]`                                                                                                                   | 0 (los `0` son válidos: clientes recién dados de alta) | IBM |
| 7  | `PhoneService`       | `category`          | Indica si el cliente tiene contratado el servicio telefónico.                                                             | `{Yes, No}`                                                                                                                 | 0                                              | IBM    |
| 8  | `MultipleLines`      | `category`          | Indica si el cliente tiene múltiples líneas telefónicas activas.                                                          | `{No phone service, No, Yes}` — `No phone service` es condicional a `PhoneService = No`.                                    | 0                                              | IBM    |
| 9  | `InternetService`    | `category`          | Tipo de servicio de internet contratado.                                                                                  | `{DSL, Fiber optic, No}`                                                                                                    | 0                                              | IBM    |
| 10 | `OnlineSecurity`     | `category`          | Indica si el cliente tiene contratado el complemento de seguridad en línea.                                               | `{No internet service, No, Yes}` — `No internet service` condicional a `InternetService = No`.                              | 0                                              | IBM    |
| 11 | `OnlineBackup`       | `category`          | Indica si el cliente tiene contratado el complemento de _backup_ en línea.                                                | `{No internet service, No, Yes}`                                                                                            | 0                                              | IBM    |
| 12 | `DeviceProtection`   | `category`          | Indica si el cliente tiene contratado el complemento de protección de dispositivos.                                       | `{No internet service, No, Yes}`                                                                                            | 0                                              | IBM    |
| 13 | `TechSupport`        | `category`          | Indica si el cliente tiene contratado el complemento de soporte técnico premium.                                          | `{No internet service, No, Yes}`                                                                                            | 0                                              | IBM    |
| 14 | `StreamingTV`        | `category`          | Indica si el cliente tiene contratado el complemento de _streaming_ de TV.                                                | `{No internet service, No, Yes}`                                                                                            | 0                                              | IBM    |
| 15 | `StreamingMovies`    | `category`          | Indica si el cliente tiene contratado el complemento de _streaming_ de películas.                                         | `{No internet service, No, Yes}`                                                                                            | 0                                              | IBM    |
| 16 | `Contract`           | `category` ordinal  | Modalidad de contrato.                                                                                                    | `{Month-to-month, One year, Two year}`                                                                                      | 0                                              | IBM    |
| 17 | `PaperlessBilling`   | `category`          | Indica si la facturación es electrónica (`Yes`) o en papel (`No`).                                                        | `{Yes, No}`                                                                                                                 | 0                                              | IBM    |
| 18 | `PaymentMethod`      | `category`          | Método de pago utilizado por el cliente.                                                                                  | `{Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)}`                                       | 0                                              | IBM    |
| 19 | `MonthlyCharges`     | `float32`           | Cargo mensual facturado al cliente (en USD).                                                                              | `[18.25, 118.75]`                                                                                                           | 0                                              | IBM    |
| 20 | `TotalCharges`       | `float32`           | Suma total facturada al cliente desde el alta (en USD).                                                                   | `[18.80, 8 684.80]` aproximadamente.                                                                                        | **11** filas con `" "` (cliente con `tenure = 0`). Tratar como `NaN`. | IBM |
| 21 | `Churn` _(target)_   | `category` binaria  | **Variable objetivo.** Indica si el cliente canceló el servicio durante el último mes.                                    | `{Yes, No}` — codificación interna: `Yes = 1`, `No = 0`.                                                                    | 0                                              | IBM    |

---

## 3. Reglas de integridad y dependencias entre variables

Existen relaciones condicionales entre algunas variables que deben respetarse durante la limpieza:

| Regla                                                                                                                | Validación implementada en `src/churnlens/data/schema.py`                |
|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Si `PhoneService = No` → `MultipleLines = No phone service`.                                                         | `pa.Check` cruzado entre ambas columnas.                                  |
| Si `InternetService = No` → `OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies` = `No internet service`. | `pa.Check` cruzado.                                                       |
| `customerID` debe ser único.                                                                                         | `pa.Check.unique()`.                                                      |
| `tenure ≥ 0`.                                                                                                        | `pa.Check.greater_than_or_equal_to(0)`.                                   |
| `MonthlyCharges` y `TotalCharges` deben ser numéricos ≥ 0.                                                            | `pa.Check.greater_than_or_equal_to(0)`.                                   |
| `Churn ∈ {Yes, No}`.                                                                                                  | `pa.Check.isin(["Yes", "No"])`.                                           |

---

## 4. Variable objetivo — análisis preliminar

| Métrica                                  | Valor      |
|------------------------------------------|------------|
| Total de filas                           | 7 043      |
| Filas con `Churn = Yes`                  | 1 869      |
| Filas con `Churn = No`                   | 5 174      |
| Tasa de churn (clase positiva)           | **26.54 %** |
| Razón de desbalance (negativos:positivos)| ~2.77 : 1  |

> **Implicación práctica:** el dataset es **moderadamente desbalanceado**. No se requiere _resampling_ agresivo, pero sí elegir métricas robustas (PR-AUC, F1) y considerar `class_weight='balanced'` o calibración de _threshold_ durante el modelado.

---

## 5. Convenciones aplicadas en el proyecto

- **Codificación de la variable objetivo:** internamente se mantiene el _string_ original (`"Yes"/"No"`) durante la carga y se traduce a `int` (`1/0`) en la capa de modelado, no en la capa de datos.
- **Tipos pandas:**
  - Categóricas → `pd.Categorical` (memoria-eficiente, ordenable cuando aplica).
  - `Contract` se carga como categórica **ordenada**: `Month-to-month < One year < Two year`.
  - `tenure` → `int16` (suficiente para el rango `[0, 72]`).
  - `MonthlyCharges`, `TotalCharges` → `float32` (precisión suficiente, ahorra ~50 % de memoria vs `float64`).
- **Valor faltante en `TotalCharges`:** se reemplaza el carácter `" "` por `NaN` durante la carga; no se imputa en esta fase (la imputación es decisión de la Fase 2).

---

## 6. Referencias internas y externas

- Origen, licencia y procedencia del dataset → ver [`data_definition.md`](data_definition.md).
- Reglas de validación implementadas → ver [`src/churnlens/data/schema.py`](../../src/churnlens/data/schema.py).
- Reporte preliminar de calidad de datos → ver [`data_quality_report.md`](data_quality_report.md).
- _Dataset original_: IBM Cognos Analytics / Kaggle _Telco Customer Churn_ — https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

<sub>_Documento del entregable de Fase 1 — Diplomado MLDS, Universidad Nacional de Colombia._</sub>
