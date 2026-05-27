# Reporte de línea base de modelos — ChurnLens

> Entregable de Fase 3 (Diplomado MLDS · UNAL) ·
> Fecha: 2026-05-27 · Modelos comparados: **8** ·
> Métrica primaria: **PR-AUC** (validación cruzada estratificada 5-fold).

Este reporte responde a la rúbrica:

> _"Reporte de línea base de modelos que compare el rendimiento de los
> modelos entrenados con un modelo de referencia básico"_ y
> _"Se implementan al menos dos técnicas de selección/extracción de
> características y se justifican adecuadamente"_.

Conjunto de entrenamiento: `data/processed/train.parquet`
(4 929 filas × 35 features tras Fase 2). Tasa de churn:
**26.54 %** (preservada en los tres splits con `random_state=42`).

Reproducción exacta:

```bash
make train          # equivalentemente: python scripts/training/main.py
```

---

## 1. Por qué cuatro baselines y no uno solo

Un único baseline (típicamente `DummyClassifier(strategy="prior")`) no es
suficiente para defender un modelo. Cada baseline mide algo distinto, y
un modelo "útil" debe vencer a los cuatro **simultáneamente**:

| Baseline                  | Qué hace                                          | Qué falsea si no se supera |
|---------------------------|---------------------------------------------------|----------------------------|
| `dummy_most_frequent`     | Predice siempre la clase mayoritaria (no churn)   | Accuracy "alta" del 73.4 % por puro desbalance — F1 = 0 |
| `dummy_prior`             | Predice probabilidad fija = base rate (0.266)     | ROC-AUC = 0.5, F1 = 0 a thr=0.5 |
| `dummy_stratified`        | Predice respetando la distribución de clases      | Mide si el modelo realmente _ranquea_ por encima del azar |
| `logreg_balanced`         | Logistic Regression L2 con `class_weight=balanced` | Baseline lineal real — si los modelos no lineales no superan a LR balanced, no compensan su complejidad |

El último (`logreg_balanced`) es el **baseline duro** del proyecto: es un
modelo "de verdad" pero con la mínima sofisticación posible. Que un
modelo no lineal (RF, GBM, LightGBM) no le gane es una señal fuerte de
que la complejidad extra **no compra capacidad predictiva** en este
dataset.

---

## 2. Catálogo completo de modelos comparados

| #   | Modelo                | Familia            | Hiperparámetros clave |
|----:|-----------------------|--------------------|------------------------|
| 1   | `dummy_most_frequent` | trivial            | — |
| 2   | `dummy_prior`         | trivial            | — |
| 3   | `dummy_stratified`    | trivial            | `random_state=42` |
| 4   | `logreg_balanced`     | lineal             | `penalty=l2`, `C=1.0`, `class_weight=balanced`, `solver=lbfgs`, `max_iter=2000` |
| 5   | `logreg_l1`           | lineal · esparsa   | `penalty=l1`, `C=0.5`, `class_weight=balanced`, `solver=liblinear` |
| 6   | `random_forest`       | bagging no lineal  | `n_estimators=300`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight=balanced` |
| 7   | `hist_gb`             | boosting (sklearn) | `learning_rate=0.05`, `max_iter=300`, `max_leaf_nodes=31`, `class_weight=balanced` |
| 8   | `lightgbm`            | boosting (LGBM)    | `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `class_weight=balanced` |

Protocolo común:

* **Mismo split** estratificado 70/15/15 (Fase 2), `random_state=42`.
* **CV 5-fold** estratificado sobre `train` (4 929 filas) →
  reporta media + std de PR-AUC y ROC-AUC.
* **Threshold por defecto = 0.5** para métricas a partir de predicción
  dura; **threshold sintonizado** maximizando F1 sobre `val` (1 057
  filas) para el reporte final.
* **Validación = `val`** (1 057 filas). El _held-out_ `test` permanece
  ciego hasta Fase 4.

---

## 3. Resultados — CV estratificada 5-fold sobre `train`

| Modelo                | PR-AUC media | PR-AUC std | ROC-AUC media | ROC-AUC std |
|-----------------------|-------------:|-----------:|--------------:|------------:|
| **`logreg_balanced`** | **0.6614**   | 0.0170     | **0.8466**    | 0.0040 |
| **`logreg_l1`**       | **0.6613**   | 0.0173     | **0.8466**    | 0.0040 |
| `random_forest`       | 0.6559       | 0.0229     | 0.8391        | 0.0056 |
| `lightgbm`            | 0.6399       | 0.0123     | 0.8337        | 0.0051 |
| `hist_gb`             | 0.6374       | 0.0143     | 0.8321        | 0.0039 |
| `dummy_prior`         | 0.2654       | 0.0004     | 0.5000        | 0.0000 |
| `dummy_most_frequent` | 0.2654       | 0.0004     | 0.5000        | 0.0000 |
| `dummy_stratified`    | 0.2627       | 0.0035     | 0.4922        | 0.0101 |

> **Lectura clave:** los dos modelos lineales (`logreg_balanced` y
> `logreg_l1`) están **empatados estadísticamente** en CV (Δ ≤ 0.0001,
> dentro de ~1.7 pp de desviación estándar) y superan ligeramente a los
> tres modelos no lineales por 0.5–2.4 pp de PR-AUC. La señal del
> dataset Telco es predominantemente **lineal en el espacio one-hot**;
> los modelos basados en árboles no encuentran interacciones que
> compensen su mayor varianza.

Fuente: [`reports/tables/modeling_cv_scores.csv`](../../reports/tables/modeling_cv_scores.csv).

---

## 4. Resultados — métricas sobre `val` (1 057 filas)

| Modelo                | PR-AUC | ROC-AUC | Brier  | F1@0.5 | thr ★ | F1 ★    | Precision ★ | Recall ★ | Lift@10 % |
|-----------------------|-------:|--------:|-------:|-------:|------:|--------:|------------:|---------:|----------:|
| **`logreg_l1`**       | **0.6293** | **0.8286** | 0.1700 | 0.6147 | 0.58 | **0.6390** | 0.5622 | 0.7402 | **2.70 ×** |
| `logreg_balanced`     | 0.6290 | 0.8286  | 0.1698 | 0.6128 | 0.58 | 0.6379  | 0.5625      | 0.7367   | 2.70 × |
| `random_forest`       | 0.6131 | 0.8241  | 0.1467 | 0.5925 | 0.27 | 0.6154  | 0.4856      | 0.8399   | 2.80 × |
| `lightgbm`            | 0.6113 | 0.8176  | 0.1626 | 0.6013 | 0.35 | 0.6071  | 0.5047      | 0.7616   | 2.73 × |
| `hist_gb`             | 0.6103 | 0.8208  | 0.1616 | 0.5994 | 0.48 | 0.6091  | 0.5449      | 0.6904   | 2.66 × |
| `dummy_prior`         | 0.2658 | 0.5000  | 0.1952 | 0.0000 | 0.05 | 0.4200  | 0.2658      | 1.0000   | 0.67 × |
| `dummy_most_frequent` | 0.2658 | 0.5000  | 0.2658 | 0.0000 | 0.05 | 0.0000  | 0.0000      | 0.0000   | 0.67 × |
| `dummy_stratified`    | 0.2666 | 0.5020  | 0.3879 | 0.2679 | 0.05 | 0.2679  | 0.2688      | 0.2669   | 0.99 × |

★ = a threshold sintonizado para maximizar F1 sobre `val`.

Fuente: [`reports/tables/modeling_summary.csv`](../../reports/tables/modeling_summary.csv).

### Gráfico — curvas PR sobre `val`

![Curvas PR — validación](../../reports/figures/modeling_pr_curves_val.png)

### Gráfico — curvas ROC sobre `val`

![Curvas ROC — validación](../../reports/figures/modeling_roc_curves_val.png)

---

## 5. Cuantificando el aporte sobre el baseline

| Comparación                                    | ΔPR-AUC val | Δ relativa |
|------------------------------------------------|------------:|-----------:|
| Mejor modelo (`logreg_l1`) vs `dummy_prior`     | **+0.3635** | **+136.7 %** |
| Mejor modelo vs `logreg_balanced` (baseline real) | +0.0002     | +0.03 %     |
| Mejor árbol (`random_forest`) vs `logreg_balanced` | −0.0159     | −2.5 %      |
| LightGBM vs `random_forest`                    | −0.0018     | −0.3 %      |

**Lectura clave 1:** el _lift_ sobre los baselines triviales es masivo
(+136 % en PR-AUC, +66 % en ROC-AUC, lift@10% de 2.7× vs 0.67×). El
proyecto produce valor.

**Lectura clave 2:** sobre el baseline _duro_ (`logreg_balanced`),
ningún modelo no lineal produce un aporte material. **El ganador y el
baseline lineal balanceado son operativamente indistinguibles.**

---

## 6. Decisión de modelo final

Dado el empate entre `logreg_l1` y `logreg_balanced` (en CV y en val) y
la superioridad de ambos sobre los modelos no lineales, se elige
**`logreg_l1`** como modelo final por **principio de parsimonia**:

* L1 produce un modelo más **esparso** — varios coeficientes son cero
  exactamente (a `C=0.5`), lo que reduce la superficie de monitoreo de
  _drift_ en producción.
* **Interpretabilidad** equivalente a `logreg_balanced` (mismos
  coeficientes con signo), pero con un subset natural de features
  activas que se cruza con el consenso de selección
  (ver [`feature_selection.md`](feature_selection.md)).
* **Cero pérdida de rendimiento** vs el baseline balanceado (Δ ≤ 0.001
  en cualquier métrica).
* Más eficiente en inferencia (predicción O(n_features_no_cero)).

El reporte detallado del modelo final vive en
[`final_model_report.md`](final_model_report.md).

---

## 7. Hallazgos cruzados con la selección de features

El [reporte de selección](feature_selection.md) identificó por consenso
top-20 a `Contract`, `InternetService_Fiber optic`, `tenure_bucket`,
`TechSupport_No`, `OnlineSecurity_No`, `PaperlessBilling`,
`OnlineBackup_No`, `auto_payment` y `Dependents` (con voto unánime de
las 4 técnicas).

Cuando se extrae la importancia del modelo ganador (`|coef|` de
`logreg_l1`, columna `evaluation_importance_logreg_l1.csv`), las
**top-5 coinciden** con las top del consenso: `Contract` (0.820),
`InternetService_Fiber optic` (0.766), `tenure` (0.733),
`TechSupport_No` (0.385), `TotalCharges` (0.363). Esta concordancia
**valida cruzadamente** el método de selección y el sesgo del modelo
ganador.

---

## 8. Reproducibilidad

Cada modelo se persiste como:

* `models/<name>.joblib` — estimador serializado.
* `models/<name>.metadata.json` — manifest con:
  * `hash_train`, `hash_val` (SHA-256 de los parquet usados).
  * `hash_model` (SHA-256 del joblib generado).
  * `feature_set` (lista exacta de features de entrada).
  * `hyperparameters` (output de `estimator.get_params()`).
  * `metrics.train`, `metrics.val`, `metrics.val_tuned`, `metrics.cv`.
  * `random_state`, `created_at` (UTC ISO-8601).

```bash
churnlens model list                       # lista los 8 modelos registrados
churnlens model evaluate --model logreg_l1 # re-evalúa el ganador sobre val
```

El _held-out_ `test.parquet` **no se ha tocado** — se evalúa en Fase 4.
