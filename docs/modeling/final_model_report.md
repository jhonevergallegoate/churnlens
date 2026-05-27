# Reporte del modelo final — ChurnLens

> Entregable de Fase 3 (Diplomado MLDS · UNAL) ·
> Fecha: 2026-05-27 · Modelo seleccionado: **`logreg_l1`**
> (Logistic Regression con penalización L1, `class_weight=balanced`).
>
> Métrica primaria: **PR-AUC sobre `val`** = **0.6293**.
> Threshold operativo sintonizado: **0.58** (maximiza F1).

Este reporte responde a la rúbrica:

> _"Reporte del modelo final que describa en detalle el modelo
> seleccionado y sus resultados de evaluación."_

Modelo persistido en:

* [`models/logreg_l1.joblib`](../../models/logreg_l1.joblib) — estimador serializado.
* [`models/logreg_l1.metadata.json`](../../models/logreg_l1.metadata.json) — manifest auditable.

---

## 1. Identidad y propósito

| Campo                  | Valor |
|------------------------|-------|
| Nombre lógico          | `logreg_l1` |
| Algoritmo              | `sklearn.linear_model.LogisticRegression` |
| Familia                | Modelo lineal generalizado, regularización L1, _class-weight balanced_ |
| Propósito              | Probabilidad de cancelación voluntaria (churn) en el próximo ciclo de facturación |
| Features de entrada    | 35 (set completo post-`ColumnTransformer`) |
| Variable objetivo      | `Churn` binaria (1 = cancela, 0 = retiene) |
| Tasa base de positivos | 26.54 % (4 929 train · 1 057 val · 1 057 test) |

El elegir el modelo más simple que iguala el rendimiento óptimo
([baseline_models.md §6](baseline_models.md)) responde a tres
restricciones operativas del proyecto:

1. **Interpretabilidad** — los coeficientes son legibles para
   _stakeholders_ no técnicos (equipo de retención).
2. **Esparsidad** — L1 fuerza coeficientes a cero exactamente; reduce
   la superficie de _data drift_ a monitorear.
3. **Eficiencia de inferencia** — orden ms-por-cliente sin GPU; trivial
   para una API REST en Fase 4.

---

## 2. Hiperparámetros

| Parámetro          | Valor          | Justificación |
|--------------------|----------------|---------------|
| `penalty`          | `"l1"`         | Esparsidad — corta features no informativas. |
| `C`                | `0.5`          | Regularización moderada; valores menores apagan señales débiles, mayores no esparsan suficiente. |
| `solver`           | `"liblinear"`  | El único solver de sklearn que soporta L1 con `class_weight`. |
| `class_weight`     | `"balanced"`   | Compensa el 26.5 % de positivos sin re-sampleo (que rompería el `random_state` reproducible). |
| `max_iter`         | `2000`         | Suficiente para convergencia limpia. |
| `random_state`     | `42`           | Semilla global del proyecto (`settings.random_seed`). |

Los hiperparámetros se persisten en
`models/logreg_l1.metadata.json` bajo la clave `hyperparameters` para
auditoría.

---

## 3. Protocolo de evaluación

* **Entrenamiento:** `data/processed/train.parquet` (4 929 × 36).
* **Validación principal:** `data/processed/val.parquet` (1 057 × 36).
* **Test held-out:** `data/processed/test.parquet` — **no se toca en
  esta fase**, queda para Fase 4.
* **CV durante selección:** 5-fold estratificado sobre `train`
  (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
* **Métrica primaria:** PR-AUC (`average_precision_score`) — robusta al
  desbalance moderado del dataset.
* **Métricas secundarias:** ROC-AUC, F1, precision, recall, accuracy,
  Brier score (calibración), lift@10 % (uso operativo).
* **Threshold operativo:** se sintoniza maximizando F1 sobre `val`
  con una grilla de 0.01 entre 0.05 y 0.95 (91 puntos).

---

## 4. Resultados sobre `val`

### 4.1 Métricas escalares

| Métrica           | Valor @ thr = 0.5 | Valor @ thr = 0.58 (sintonizado) |
|-------------------|------------------:|----------------------------------:|
| **PR-AUC**        | **0.6293**        | **0.6293** _(invariante al threshold)_ |
| **ROC-AUC**       | **0.8286**        | 0.8286 |
| Brier             | 0.1700            | 0.1700 |
| F1                | 0.6147            | **0.6390** |
| Precision         | 0.5106            | 0.5622 |
| Recall            | 0.7722            | 0.7402 |
| Accuracy          | 0.7427            | **0.7777** |
| Positive rate     | 40.2 %            | 35.0 % |
| Base rate         | 26.6 %            | 26.6 % |
| **Lift @ top 10 %** | **2.70 ×**      | 2.70 × |

Interpretación operativa del threshold sintonizado:

* Se etiqueta como "alto riesgo" al **35 %** de los clientes (vs el
  26.6 % real). Para un equipo de retención con presupuesto limitado,
  esto significa que **el modelo permite enfocar recursos sobre 1.3×
  más clientes que el _ground truth_ y capturar el 74 % de las
  cancelaciones reales** (recall).
* De cada 100 clientes flaggeados, 56 cancelarían (precision). Eso es
  **2.1× el _hit rate_ azar** (26.6 %).
* En el top-10 % de mayor probabilidad la tasa real de churn es **2.7×
  la base** — útil para programas premium de retención.

Fuente: [`reports/tables/evaluation_summary_logreg_l1.json`](../../reports/tables/evaluation_summary_logreg_l1.json).

### 4.2 Curvas

**Precision-Recall** (área = **0.629** vs base rate 0.266):

![PR curve · logreg_l1 · val](../../reports/figures/evaluation_pr_logreg_l1.png)

**ROC** (AUC = **0.829**):

![ROC curve · logreg_l1 · val](../../reports/figures/evaluation_roc_logreg_l1.png)

**Threshold sweep** — precision / recall / F1 vs threshold (línea
vertical = 0.58, threshold sintonizado):

![Threshold sweep · logreg_l1 · val](../../reports/figures/evaluation_threshold_logreg_l1.png)

**Matriz de confusión** a threshold sintonizado (`thr = 0.58`):

![Confusion matrix · logreg_l1 · val](../../reports/figures/evaluation_confusion_logreg_l1.png)

**Calibración** (curva de fiabilidad, 10 bins por cuantiles):

![Calibration · logreg_l1 · val](../../reports/figures/evaluation_calibration_logreg_l1.png)

> El Brier score de **0.170** está dentro del rango aceptable para un
> clasificador con `class_weight=balanced` (que tiende a sobre-estimar
> la probabilidad positiva). En Fase 4 se evaluará `CalibratedClassifierCV`
> isotónico si el negocio prefiere probabilidades estrictas en lugar de
> rankings.

### 4.3 Tablas crudas

* **Threshold sweep completo** (91 puntos):
  [`reports/tables/evaluation_threshold_sweep_logreg_l1.csv`](../../reports/tables/evaluation_threshold_sweep_logreg_l1.csv).
* **Importancia de features** (|coef|):
  [`reports/tables/evaluation_importance_logreg_l1.csv`](../../reports/tables/evaluation_importance_logreg_l1.csv).
* **Resumen JSON**:
  [`reports/tables/evaluation_summary_logreg_l1.json`](../../reports/tables/evaluation_summary_logreg_l1.json).

---

## 5. Importancia de features del ganador

Las top-10 features por valor absoluto de coeficiente:

| #  | Feature                       | `|coef|` |
|---:|-------------------------------|---------:|
| 1  | `Contract`                    | 0.8200 |
| 2  | `InternetService_Fiber optic` | 0.7659 |
| 3  | `tenure`                      | 0.7334 |
| 4  | `TechSupport_No`              | 0.3850 |
| 5  | `TotalCharges`                | 0.3629 |
| 6  | `PaperlessBilling`            | 0.3557 |
| 7  | `PaymentMethod_Mailed check`  | 0.3501 |
| 8  | `auto_payment`                | 0.3456 |
| 9  | `has_phone`                   | 0.3009 |
| 10 | `OnlineSecurity_No`           | 0.2819 |

Coeficientes apagados a 0 por L1 (esparsidad lograda — **8 de 35**
features fueron eliminadas automáticamente): `MonthlyCharges`,
`services_count`, `has_internet`, `Partner`, `InternetService_No`,
`OnlineBackup_Yes`, `DeviceProtection_No`, `DeviceProtection_Yes`,
`TechSupport_Yes`, `StreamingTV_No`, `StreamingMovies_No`,
`PaymentMethod_Credit card (automatic)`, `MultipleLines_Yes`.

> **Lectura cruzada:** las 5 features con mayor coeficiente coinciden
> con el top-5 del **consenso** de selección
> ([feature_selection.md](feature_selection.md) §3) — la decisión del
> filtro y la del modelo apuntan al mismo subconjunto.

![Importancia de features · logreg_l1](../../reports/figures/evaluation_importance_logreg_l1.png)

### Lectura cualitativa de los coeficientes

* **Contract** (categoría ordinal `Month-to-month` → `One year` → `Two year`):
  signo negativo grande — contratos largos reducen la probabilidad de
  churn. Consistente con el EDA (Month-to-month: 42.7 %; Two-year: 2.8 %).
* **InternetService_Fiber optic**: signo positivo grande — los
  clientes con fibra óptica churnean más, probablemente por precio
  premium percibido o por _competencia local_ de ISPs nicho.
* **tenure / TotalCharges**: ambos correlacionan con permanencia
  histórica — coef negativo → mayor antigüedad reduce churn.
* **TechSupport_No, OnlineSecurity_No**: la _ausencia_ de add-ons de
  protección sube churn — _lock-in_ por bundle de servicios.
* **PaperlessBilling, PaymentMethod_Mailed check**: clientes
  digitalizados o que pagan por cheque tienen churn mayor que los de
  pago automático (signo negativo de `auto_payment`).

---

## 6. Validación cruzada (CV 5-fold sobre `train`)

* PR-AUC media = **0.6613**, std = 0.0173.
* ROC-AUC media = **0.8466**, std = 0.0040.
* Diferencia val (`0.6293`) vs CV (`0.6613`) = **−0.032 PR-AUC** —
  consistente con el menor tamaño del set de validación; no sugiere
  sobreajuste material.

Fuente: [`reports/tables/modeling_cv_scores.csv`](../../reports/tables/modeling_cv_scores.csv).

---

## 7. Limitaciones, riesgos y _next steps_

### 7.1 Limitaciones

* **Calibración imperfecta** (Brier = 0.17 con `class_weight=balanced`).
  Los rankings son fiables; las probabilidades absolutas no lo son
  tanto. Si el negocio necesita umbrales basados en costo esperado,
  pasar por `CalibratedClassifierCV(method='isotonic', cv=5)` en
  Fase 4.
* **Frontera de decisión lineal** en el espacio one-hot — si en
  iteraciones futuras incorporamos features con interacciones fuertes
  (p. ej. CLV × tenure × producto), un GBM con _feature engineering_
  más rico podría escalar más alto.
* **Dataset estático** — el Telco Customer Churn no captura
  estacionalidad, promociones ni churn por _switch_ a competidor;
  el _domain shift_ esperado es alto.

### 7.2 Riesgos operativos

* **Coeficiente positivo en `gender` ≈ 0** — el modelo es prácticamente
  ciego al género. Se valida en Fase 4 con _fairness audit_
  (`ethics_and_fairness.md`).
* **`SeniorCitizen`** — apagado por L1; no entra al modelo en la
  configuración actual. Se re-evalúa al reintroducir el set completo
  para reportes regulatorios.
* **Probabilidades binarizadas en datos test** — el threshold 0.58 se
  fijó con `val`; mover threshold con `test` configuraría _data
  leakage_. La política operativa será **fijar el threshold con val,
  evaluarlo en test, y re-tuneo solo en re-entrenamiento periódico**.

### 7.3 Próximos pasos (Fase 4)

1. **Evaluación sobre `test.parquet`** y reporte definitivo de
   métricas held-out.
2. **Calibración** con `CalibratedClassifierCV` si Brier > 0.18 en test.
3. **Fairness audit** por `gender` y `SeniorCitizen` (paridad de
   recall y precision por subgrupo).
4. **API REST + monitoreo** (`docs/architecture/solution_architecture.md`):
   _drift_ de features, _data quality_, alertas por degradación de
   PR-AUC.
5. **Re-entrenamiento periódico** con nueva data tras el primer mes en
   producción.

---

## 8. Reproducibilidad bit-exacta

```bash
make phase3                                  # Fase 2 + selección + train + eval
# o paso a paso
make preprocess
make features
make train                                   # 8 modelos con CV 5-fold
make evaluate                                # ganador → reports/figures + tables
churnlens model evaluate --model logreg_l1   # spot-check rápido
```

El manifest del modelo guardado incluye los hashes SHA-256 de los
parquet usados como entrada, del joblib generado, y los hiperparámetros
exactos. Cualquier ejecución posterior con la misma semilla
(`settings.random_seed=42`), el mismo `train.parquet` y los mismos
hiperparámetros produce un modelo **byte-equivalente**.
