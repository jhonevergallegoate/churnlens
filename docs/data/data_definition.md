# Definición de los datos — ChurnLens

> **Propósito:** documentar el origen, la licencia, la ubicación, la estructura y los procedimientos de carga, transformación y persistencia de los datos utilizados en el proyecto.

---

## 1. Origen de los datos

### 1.1 Nombre y procedencia

- **Nombre oficial:** _Telco Customer Churn_
- **Publicador original:** **IBM Cognos Analytics** (IBM Sample Data Sets).
- **Réplica pública usada:** _mirror_ oficial mantenido por IBM en GitHub →
  `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Réplica alterna (Kaggle):** _Telco Customer Churn_ por _BlastChar_ →
  `https://www.kaggle.com/datasets/blastchar/telco-customer-churn`

### 1.2 Tipo de fuente

- **Naturaleza:** archivo plano CSV de **un único snapshot**.
- **Periodicidad:** _no se actualiza_ (es un dataset académico estático).
- **Granularidad:** una fila por cliente único.
- **Ventana temporal cubierta:** valores aproximados a un punto en el tiempo; la etiqueta `Churn` se refiere al **último mes** previo al _snapshot_.

### 1.3 Cómo se obtiene

La descarga se realiza de forma automatizada vía HTTPS por el módulo `churnlens.data.loader` o el script `scripts/data_acquisition/main.py`. La URL es configurable mediante la variable de entorno `CHURNLENS_DATA_URL` (definida en `.env.example`).

```bash
# Vía CLI
churnlens data download

# Vía Makefile
make data-download

# Vía script TDSP
python scripts/data_acquisition/main.py
```

### 1.4 Licencia y atribución

- Dataset propiedad de **IBM Corp.**
- Uso permitido para fines **académicos, de investigación y no comerciales**.
- Atribución obligatoria al publicador original en cualquier publicación derivada.
- Réplica en Kaggle bajo términos de _Kaggle's open data terms_.

> **Atribución sugerida:**
> _Dataset "Telco Customer Churn" © IBM Corporation. Usado bajo términos académicos a través del mirror público IBM/telco-customer-churn-on-icp4d._

---

## 2. Especificación de los scripts de carga

### 2.1 Punto de entrada principal — script TDSP

| Ruta                                          | Propósito                                                                    |
|-----------------------------------------------|------------------------------------------------------------------------------|
| `scripts/data_acquisition/main.py`            | Punto de entrada exigido por la rúbrica TDSP. Orquesta descarga + validación. |

Ejecutable con:

```bash
python scripts/data_acquisition/main.py
```

Acepta los siguientes argumentos opcionales:

| Flag                  | Descripción                                          | Default                              |
|-----------------------|------------------------------------------------------|--------------------------------------|
| `--url URL`           | URL fuente del CSV.                                  | Variable `CHURNLENS_DATA_URL`.       |
| `--output PATH`       | Ruta destino del CSV crudo.                          | `data/raw/telco_customer_churn.csv`. |
| `--force`             | Re-descarga aunque el archivo ya exista.             | `False`.                              |
| `--no-validate`       | Omite la validación con Pandera.                     | `False`.                              |
| `--verbose / --quiet` | Controla el nivel de _logging_.                      | INFO.                                 |

### 2.2 CLI unificada (consumo programático)

Las mismas operaciones están disponibles a través de la CLI principal:

```bash
churnlens data download   [--force]
churnlens data validate
churnlens data summary
churnlens data hash       # imprime MD5 del archivo crudo
```

### 2.3 Módulos del paquete `churnlens`

| Módulo                              | Responsabilidad                                                                |
|-------------------------------------|--------------------------------------------------------------------------------|
| `churnlens.config`                  | Configuración tipada con Pydantic Settings, hidratada desde `.env`.            |
| `churnlens.logger`                  | Logger estructurado (`structlog`) con formato consola / JSON.                   |
| `churnlens.data.loader`             | Descarga, carga y persistencia parquet.                                         |
| `churnlens.data.schema`             | Esquema Pandera del dataset crudo + checks cruzados.                            |
| `churnlens.data.validators`         | Validaciones de integridad (unicidad, dependencias, dominios).                  |
| `churnlens.utils.hashing`           | Cálculo y verificación de hashes MD5/SHA-256.                                   |
| `churnlens.cli`                     | CLI principal con `typer`.                                                      |

---

## 3. Rutas y estructura del almacenamiento local

### 3.1 Convención de carpetas (`data/`)

| Carpeta             | Propósito                                                                                  | ¿Se versiona en git? |
|---------------------|---------------------------------------------------------------------------------------------|----------------------|
| `data/raw/`         | Datos **inmutables** tal como fueron descargados desde la fuente original.                  | ❌ — solo `.gitkeep`. |
| `data/interim/`     | Resultados intermedios reproducibles (parquet validado, casteo de tipos).                   | ❌                    |
| `data/processed/`   | Datasets listos para modelar (post-_feature engineering_, Fase 2+).                         | ❌                    |
| `data/external/`    | Fuentes externas adicionales (no aplica en Fase 1).                                         | ❌                    |

> Esta política protege la privacidad y mantiene el repositorio liviano. La reproducibilidad se garantiza ejecutando `make data`.

### 3.2 Archivos generados por la Fase 1

| Archivo                                          | Contenido                                                              |
|--------------------------------------------------|------------------------------------------------------------------------|
| `data/raw/telco_customer_churn.csv`              | CSV crudo tal como se descarga del _mirror_ de IBM.                    |
| `data/interim/telco_customer_churn.parquet`      | Dataset validado y con tipos estables (parquet con compresión `snappy`). |
| `data/raw/.checksums.json`                       | Hashes MD5 + SHA-256 del archivo crudo (auditoría de integridad).      |

### 3.3 Archivos generados por la Fase 2

| Archivo                                          | Contenido                                                              |
|--------------------------------------------------|------------------------------------------------------------------------|
| `data/processed/train.parquet`                   | 4 929 filas — partición de entrenamiento (70 %).                       |
| `data/processed/val.parquet`                     | 1 057 filas — partición de validación (15 %).                          |
| `data/processed/test.parquet`                    | 1 057 filas — partición de prueba (15 %).                              |
| `data/processed/preprocessor.joblib`             | `ColumnTransformer` de scikit-learn ajustado **solo** a `train`.       |
| `data/processed/feature_names.json`              | Nombres de columna post-transformación (35 features).                  |
| `data/processed/metadata.json`                   | Shapes, tasa de positivos por split, semilla, fecha de generación.     |
| `reports/figures/eda_*.png`                      | 9 figuras del análisis exploratorio.                                    |
| `reports/tables/eda_*.csv`                       | 4 tablas (resumen numérico, categórico, V de Cramér, distribución target).|

---

## 4. Esquema técnico del archivo crudo

### 4.1 Formato

- **Tipo:** archivo plano CSV (UTF-8, sin BOM).
- **Separador:** coma (`,`).
- **Salto de línea:** `\n`.
- **Encabezado:** primera fila (21 nombres de columna).
- **Comillas:** ninguna en el archivo original.
- **Tamaño aproximado:** ~ 977 KB.
- **Filas:** 7 043 (sin contar el encabezado).
- **Columnas:** 21 (ver [`data_dictionary.md`](data_dictionary.md)).

### 4.2 Hash de referencia (huella digital)

Al descargar el archivo, el _loader_ guarda el hash en `data/raw/.checksums.json` para que cualquier ejecución posterior pueda verificar la integridad.

```jsonc
// data/raw/.checksums.json (ejemplo del payload escrito por el loader)
{
  "telco_customer_churn.csv": {
    "md5":    "<hash-md5-calculado-al-descargar>",
    "sha256": "<hash-sha256-calculado-al-descargar>",
    "bytes":  977501,
    "downloaded_at": "2026-05-14T20:53:00Z"
  }
}
```

> En la **primera ejecución** se registra el hash observado. En ejecuciones posteriores, el _loader_ compara y emite un _warning_ si difiere — esto previene corrupciones silenciosas y _mirror drift_.

---

## 5. Procedimientos de transformación y limpieza

El pipeline aplicado durante la carga es **conservador**: solo se ejecutan transformaciones reversibles y bien justificadas. La limpieza profunda se reserva para la Fase 2.

### 5.1 Pipeline implementado en Fase 1

```
1. Descarga del CSV → data/raw/
2. Cálculo y registro de hash MD5/SHA-256
3. Lectura con pandas (encoding='utf-8')
4. Conversión de columna TotalCharges:
     - Strip whitespace
     - Conversión "" → NaN
     - Cast a float32
5. Casting de tipos (ver tabla §5.2)
6. Validación con esquema Pandera (lazy=True)
7. Persistencia a data/interim/*.parquet
```

### 5.2 Tabla de _casting_ de tipos

| Columna                                                              | Tipo origen (pandas default) | Tipo final cargado       |
|----------------------------------------------------------------------|------------------------------|--------------------------|
| `customerID`                                                         | `object`                     | `string`                 |
| `gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn` | `object`                     | `category` (binaria)     |
| `SeniorCitizen`                                                      | `int64`                      | `int8`                   |
| `tenure`                                                             | `int64`                      | `int16`                  |
| `MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies` | `object` | `category`               |
| `Contract`                                                           | `object`                     | `category` **ordenada**  |
| `PaymentMethod`                                                      | `object`                     | `category`               |
| `MonthlyCharges`                                                     | `float64`                    | `float32`                |
| `TotalCharges`                                                       | `object`                     | `float32` (con NaN)      |

### 5.3 Política de no-imputación en Fase 1

En la Fase 1 **no se imputa** ningún valor faltante. La razón:

- La imputación es una decisión de modelado que debe quedar trazada como _step_ del pipeline reproducible (Fase 2).
- Imputar antes de explorar puede ocultar señales informativas (un `TotalCharges = NaN` siempre coincide con `tenure = 0`, lo cual es _informativo_ en sí mismo).

### 5.4 Pipeline de preprocesamiento — Fase 2

A partir de la Fase 2, sobre el dataset validado se aplica un
`ColumnTransformer` reproducible (`src/churnlens/features/preprocessing.py`)
con cuatro bloques:

```
1. Numéricas continuas (`tenure`, `MonthlyCharges`, `TotalCharges` + derivadas):
     SimpleImputer(strategy="median") → StandardScaler()
2. Ordinales (`Contract`, `tenure_bucket`):
     OrdinalEncoder con orden explícito
3. Binarias (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`
   + booleanos derivados):
     OrdinalEncoder con orden fijo
4. Nominales multi-clase (`MultipleLines`, `InternetService`, 6 add-ons, `PaymentMethod`):
     OneHotEncoder(drop="first", handle_unknown="ignore")
```

El _transformer_ se **ajusta exclusivamente** sobre el conjunto de
entrenamiento (estratificado, 70 %) y se aplica a `val` (15 %) y `test`
(15 %), garantizando ausencia de _leakage_. Persiste serializado como
`preprocessor.joblib` para inferencia futura.

Antes del transformador, las **features derivadas** (`tenure_bucket`,
`services_count`, `has_internet`, `has_phone`, `auto_payment`,
`avg_monthly_spend`, `monthly_spend_gap`) se generan determinísticamente
desde el esquema validado en
`src/churnlens/features/engineering.py::add_engineered_features`.

Ver justificación de cada decisión en
[`data_summary_report.md`](data_summary_report.md) §9.

---

## 6. Base de datos de destino

En esta versión del proyecto **no se utiliza un sistema de gestión de bases de datos**. La persistencia se hace sobre el sistema de archivos local en formato _parquet_ comprimido. Esta decisión se justifica por:

- El dataset es pequeño (~7 K filas) — no se requiere SGBD.
- Parquet conserva _schema_ y tipos.
- El _flat-file approach_ facilita la reproducibilidad y portabilidad.
- Es coherente con la estructura TDSP estándar (`data/raw/`, `data/interim/`, `data/processed/`).

> En un escenario productivo real, el equivalente sería una tabla `customers_features` en un _data warehouse_ (Redshift, BigQuery, Snowflake) o un _feature store_ (Feast).

---

## 7. Reproducibilidad y trazabilidad

| Garantía                                       | Mecanismo                                                              |
|------------------------------------------------|------------------------------------------------------------------------|
| **Mismo dataset entre ejecuciones**            | Hash MD5 verificado en cada `data download`.                           |
| **Mismas dependencias**                        | `pyproject.toml` con _ranges_ acotados.                                 |
| **Misma semilla aleatoria**                    | `CHURNLENS_RANDOM_SEED=42` definido en `.env` y consumido por `config`. |
| **Mismas transformaciones**                    | Pipeline encapsulado en `churnlens.data.loader`.                       |
| **Auditoría de cambios**                       | Git + commits firmables + CI.                                          |

---

## 8. Riesgos sobre los datos

| Riesgo                                                                                 | Mitigación                                                                           |
|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| El _mirror_ de IBM cambia o desaparece.                                                | Se permite configurar `CHURNLENS_DATA_URL`; el hash detecta cualquier cambio silencioso. |
| El dataset es viejo y podría no reflejar la realidad actual del mercado.               | Se documenta en _model card_ y en la _limitation section_.                            |
| El dataset es pequeño y limita la capacidad de generalización.                         | Validación cruzada estratificada + intervalos de confianza sobre métricas (Fase 3). |
| `TotalCharges` con cadenas vacías podría romper el parseo.                              | Se trata explícitamente durante el _loader_ (paso 5.1.4).                            |

---

## 9. Referencias internas

- [`data_dictionary.md`](data_dictionary.md) — diccionario detallado por variable.
- [`data_quality_report.md`](data_quality_report.md) — reporte preliminar de calidad.
- [`data_summary_report.md`](data_summary_report.md) — reporte de resumen Fase 2.
- [`src/churnlens/data/loader.py`](../../src/churnlens/data/loader.py) — implementación del _loader_.
- [`src/churnlens/data/schema.py`](../../src/churnlens/data/schema.py) — esquema Pandera.
- [`src/churnlens/features/engineering.py`](../../src/churnlens/features/engineering.py) — features derivadas Fase 2.
- [`src/churnlens/features/preprocessing.py`](../../src/churnlens/features/preprocessing.py) — `ColumnTransformer` Fase 2.
- [`src/churnlens/features/pipeline.py`](../../src/churnlens/features/pipeline.py) — orquestador end-to-end Fase 2.
- [`src/churnlens/eda/`](../../src/churnlens/eda/) — estadísticas y visualizaciones Fase 2.
- [`scripts/data_acquisition/main.py`](../../scripts/data_acquisition/main.py) — punto de entrada TDSP Fase 1.
- [`scripts/preprocessing/main.py`](../../scripts/preprocessing/main.py) — punto de entrada TDSP Fase 2.
- [`scripts/eda/main.py`](../../scripts/eda/main.py) — punto de entrada TDSP Fase 2.

---

<sub>_Documento del entregable de Fase 1 — Diplomado MLDS, Universidad Nacional de Colombia._</sub>
