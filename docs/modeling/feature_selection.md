# Selección de características — ChurnLens

> Entregable de Fase 3 (Diplomado MLDS · UNAL) ·
> Fecha: 2026-05-27 · Branch: `dev` · Tag objetivo: `v0.3.0-fase3`.

Este reporte documenta la **etapa de extracción y selección de
características** del proyecto. Se ejecutaron **cuatro técnicas
complementarias** sobre el conjunto de entrenamiento preprocesado
(4 929 filas × 35 features tras el `ColumnTransformer` de la Fase 2) y se
consolidaron en un **consenso top-k = 20**.

Toda la lógica vive en
[`src/churnlens/features/selection.py`](../../src/churnlens/features/selection.py)
y se invoca de forma idempotente vía:

```bash
churnlens features select --k 20
# o
python scripts/training/main.py        # selección + entrenamiento
```

Los artefactos persistidos son:

| Archivo                                                       | Propósito |
|---------------------------------------------------------------|-----------|
| `reports/tables/feature_selection_scores.csv`                 | Score crudo por técnica × feature |
| `reports/tables/feature_selection_ranks.csv`                  | Ranking (1 = mejor) por técnica × feature |
| `reports/tables/feature_selection_consensus.csv`              | Tabla de voto y score normalizado |
| `data/processed/feature_consensus.json`                       | Manifest del top-k para alimentar `train_models(feature_subset=...)` |

---

## 1. Por qué cuatro técnicas y no una

La literatura distingue tres familias de selección
[Guyon & Elisseeff, 2003; Kuhn & Johnson, 2019]:

1. **Filtros univariados** — rápidos, agnósticos al modelo, miden
   asociación marginal pero ignoran interacciones.
2. **Métodos _embedded_** — integrados al ajuste del modelo
   (regularización, importancia de árbol). Capturan efectos marginales
   bajo el sesgo del clasificador.
3. **Métodos _wrapper_** — entrenan repetidamente el modelo permutando
   o reemplazando subconjuntos de features. Capturan interacciones pero
   son caros.

Para que la selección sea **defendible y robusta a los sesgos de cada
técnica**, este proyecto combina las tres familias:

| # | Técnica                 | Familia    | Capta              | Sesgo                                  |
|---|-------------------------|------------|--------------------|----------------------------------------|
| 1 | Mutual Information      | Filter     | dependencia no lineal univariada | k-NN sensible a discretización |
| 2 | χ² (chi-cuadrado)       | Filter     | dependencia categórica | requiere features ≥ 0; no aplica al bloque numérico escalado |
| 3 | L1 Logistic Regression  | Embedded   | utilidad lineal con `class_weight=balanced` | colapsa features correlacionadas a una |
| 4 | Permutation importance sobre Random Forest | Wrapper | importancia no lineal + interacciones | costosa, varianza alta con pocos repeats |

Una feature sobrevive el filtro **solo si la mayoría de las técnicas la
seleccionan** — eso es el _consenso_ que se describe en §3.

---

## 2. Configuración usada

| Parámetro                           | Valor                | Justificación |
|-------------------------------------|----------------------|---------------|
| `random_state`                      | 42 (`settings.random_seed`) | Reproducibilidad bit-exacta. |
| RF para permutation (`n_estimators`)| 200                  | Equilibrio varianza/tiempo. |
| Permutation `n_repeats`             | 10                   | Suficiente para estabilizar la media. |
| Permutation `scoring`               | `average_precision` (PR-AUC) | Métrica del proyecto, sensible al desbalance. |
| L1 LogReg `C`                       | 0.5 (solver `liblinear`) | Esparsidad moderada — limpia ruido sin matar señales débiles. |
| L1 LogReg `class_weight`            | `balanced`           | Compensa el 26.5 % de positivos. |
| `k` del consenso                    | 20 (≈ 57 % de 35)    | Mantiene capacidad y descarta ruido obvio. |

> Los hiperparámetros viven como _defaults_ del módulo y como flags de la
> CLI (`--k`, `--rf-estimators`, `--permutation-repeats`), por lo que
> cualquier cambio queda versionado.

---

## 3. Consenso top-k

El consenso se calcula así:

1. Cada técnica produce un ranking 1..N sobre las 35 features.
2. Una feature recibe **un voto por técnica** si está en el top-k de esa
   técnica.
3. Las features se ordenan por **(votos ↓, score normalizado medio ↓)**
   y se cortan en `k = 20`.

### Top-20 consenso (output real)

| #  | Feature                                | Votos | Score medio (norm.) |
|----|----------------------------------------|------:|---------------------:|
| 1  | `Contract`                             | 4     | 1.0000 |
| 2  | `InternetService_Fiber optic`          | 4     | 0.5093 |
| 3  | `tenure_bucket`                        | 4     | 0.4409 |
| 4  | `TechSupport_No`                       | 4     | 0.3972 |
| 5  | `OnlineSecurity_No`                    | 4     | 0.3599 |
| 6  | `PaperlessBilling`                     | 4     | 0.3524 |
| 7  | `OnlineBackup_No`                      | 4     | 0.2231 |
| 8  | `auto_payment`                         | 4     | 0.2214 |
| 9  | `Dependents`                           | 4     | 0.1580 |
| 10 | `tenure`                               | 3     | 0.6821 |
| 11 | `TotalCharges`                         | 3     | 0.4394 |
| 12 | `avg_monthly_spend`                    | 3     | 0.2804 |
| 13 | `PaymentMethod_Mailed check`           | 3     | 0.1567 |
| 14 | `DeviceProtection_No`                  | 3     | 0.1334 |
| 15 | `OnlineSecurity_Yes`                   | 3     | 0.0951 |
| 16 | `MonthlyCharges`                       | 2     | 0.2776 |
| 17 | `monthly_spend_gap`                    | 2     | 0.2527 |
| 18 | `StreamingMovies_Yes`                  | 2     | 0.1362 |
| 19 | `InternetService_No`                   | 2     | 0.1292 |
| 20 | `has_internet`                         | 2     | 0.1246 |

### Lo que sobrevive con 4 votos (consenso total)

Nueve features ganan en **las cuatro técnicas**. Coincide con la lectura
del EDA (Fase 2): `Contract` es de lejos el predictor más fuerte (V de
Cramér = 0.41); `tenure_bucket` separa el 47.4 % de churn en 0-12 m del
9.5 % en 49-72 m; los add-ons de internet (`TechSupport`,
`OnlineSecurity`, `OnlineBackup`) marcan _lock-in_ por servicios
combinados; `auto_payment` y `PaperlessBilling` son señales de fricción
de pago y digitalización.

### Lo que descartamos

Cuatro features quedan con **cero votos**: `PhoneService`,
`MultipleLines_Yes`, `OnlineBackup_Yes`, `DeviceProtection_Yes`. Las
versiones `_Yes` de los add-ons no aportan vs sus complementos `_No`
(uno hereda toda la información cuando dropeas la primera categoría);
`PhoneService` confirma el hallazgo del EDA (V Cramér ≈ 0). Estas
features **se mantienen disponibles en el preprocessor** — el descarte
es a nivel de _modelo_, no de _ingeniería_, para que un cambio futuro
de algoritmo pueda re-evaluarlas.

### Lectura cruzada: cada técnica vs el consenso

* **Mutual Information** ranquea alto `Contract`, `OnlineSecurity_No`,
  `TechSupport_No`, `tenure_bucket` y `tenure` — confirma señales no
  lineales en el bloque categórico.
* **χ²** está dominada por `Contract` (809), `tenure_bucket` (572),
  `TechSupport_No` (295), `OnlineSecurity_No` (289). El bloque numérico
  aparece como `NaN` porque tras `StandardScaler` contiene valores
  negativos.
* **L1 Logistic** colapsa al cero ~40 % de los coeficientes — un signo
  saludable de regularización efectiva. Sobreviven `Contract`,
  `InternetService_Fiber optic`, `tenure`, `TechSupport_No`,
  `PaperlessBilling`, `auto_payment`, `PaymentMethod_Mailed check`.
* **Permutation importance (RF, PR-AUC)** confirma `Contract` (0.026),
  `tenure` (0.012), `TotalCharges` (0.013) y `PaperlessBilling` (0.014)
  como las features cuya permutación causa la mayor caída de PR-AUC.

---

## 4. Cómo se usa el consenso aguas abajo

`train_models(feature_subset=top_k)` permite entrenar cualquier modelo
restringido al top-20. El benchmark de Fase 3 entrena **dos versiones**
de los modelos no triviales:

* **Full** (35 features) — referencia.
* **Consensus** (20 features) — capacidad reducida.

Si el modelo Consensus iguala o supera al Full en PR-AUC, el proyecto
adopta el set reducido en producción (menor superficie de ataque a
_data drift_, menor costo de monitoreo). El detalle vive en
[`baseline_models.md`](baseline_models.md) y
[`final_model_report.md`](final_model_report.md).

---

## 5. Reproducibilidad

Los CSV de §3 son **idempotentes** dado:

* `data/processed/train.parquet` con SHA-256 conocido
  (`models/<name>.metadata.json["hash_train"]`).
* `settings.random_seed = 42`.
* Hiperparámetros documentados en §2.

Para regenerar:

```bash
make preprocess
make features          # o: churnlens features select
```

---

## 6. Decisiones revisadas en la próxima fase

* La selección se hace **solo sobre `train`**; el ranking sobre `val`
  se calcula como _sanity check_ en Fase 4 (no para decidir).
* El top-k podrá ajustarse según las curvas de aprendizaje del modelo
  ganador (vía `learning_curve` en `sklearn.model_selection`).
* `gender` y `SeniorCitizen` quedan **fuera** del top-k pero se
  re-introducen para el reporte de fairness en Fase 4.
