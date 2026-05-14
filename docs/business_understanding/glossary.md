# Glosario — ChurnLens

> **Propósito:** unificar el vocabulario técnico, de negocio y metodológico empleado en el proyecto. Términos en orden alfabético; cuando aplica, se incluye el equivalente en inglés y/o español.

---

## A

- **ARPU** _(Average Revenue Per User)_ — Ingreso recurrente promedio por usuario. Para este proyecto se aproxima a la mediana de `MonthlyCharges`.
- **AUC** _(Area Under the Curve)_ — Área bajo una curva. Sin calificador suele referirse a ROC-AUC.

## B

- **Baseline** — Modelo simple usado como punto de comparación obligatorio. En este proyecto: regresión logística regularizada.
- **Boosting** — Familia de algoritmos que combina aprendices débiles secuencialmente; ejemplos: XGBoost, LightGBM.

## C

- **Calibración** — Grado de coincidencia entre las probabilidades predichas y las frecuencias observadas. Se mide con _ECE_ y diagramas de confiabilidad.
- **Churn** — Cancelación voluntaria del servicio por parte del cliente. Variable _target_ del proyecto.
- **Churn rate** — % de clientes activos al inicio del período que cancelan durante el período.
- **CRISP-DM** — Metodología clásica de _data mining_; antecesora del TDSP.
- **CV** _(Cross-Validation)_ — Validación cruzada. Para clasificación desbalanceada se prefiere _StratifiedKFold_.

## D

- **Data leakage** — Filtración accidental de información del _target_ hacia las _features_ usadas para entrenar. Riesgo crítico en este proyecto (ver R-03).
- **Disparate Impact** — Razón entre la tasa de selección de un grupo desfavorecido y un grupo favorecido para un atributo protegido. Regla 80 % de la EEOC.
- **Drift** — Desviación entre la distribución de los datos en producción y la distribución del entrenamiento. Se mide con PSI, KL-divergence, KS test.

## E

- **ECE** _(Expected Calibration Error)_ — Métrica de calibración; mide la distancia ponderada entre confianza promedio y precisión por _bin_.
- **EDA** _(Exploratory Data Analysis)_ — Análisis exploratorio de datos; primer contacto numérico y visual con el dataset.

## F

- **F1-score** — Media armónica entre _precision_ y _recall_; reportada típicamente sobre la clase minoritaria en problemas desbalanceados.
- **Fairness** — Conjunto de propiedades del modelo orientadas a evitar discriminación sobre grupos protegidos.
- **Feature engineering** — Creación de variables derivadas a partir de las originales para mejorar el _signal_ disponible al modelo.
- **Fluid Compute** — Infraestructura de Vercel basada en compartir _runtime_ entre invocaciones (referencia externa, no usada en este proyecto).

## H

- **Holdout** — Partición del dataset reservada para evaluación final; nunca tocada durante entrenamiento o _tuning_.

## L

- **Lift** — Cuántas veces mejor predice el modelo respecto a un _baseline_ aleatorio en un _bucket_ (ej. decil superior).
- **LightGBM** — Implementación eficiente de _gradient boosting_ basada en histogramas.
- **LTV** _(Lifetime Value)_ — Valor presente neto de todos los ingresos futuros esperados de un cliente.

## M

- **MLDS** — _Machine Learning and Data Science_; programa de formación de la Universidad Nacional de Colombia.
- **Model card** — Documento estándar que reporta capacidades, limitaciones, métricas y consideraciones éticas de un modelo (Mitchell et al., 2019).
- **MoSCoW** — Técnica de priorización: _Must-have_, _Should-have_, _Could-have_, _Won't-have_.
- **MRR** _(Monthly Recurring Revenue)_ — Ingreso recurrente mensual proveniente de suscripciones activas.

## O

- **OHE** _(One-Hot Encoding)_ — Técnica de codificación de variables categóricas en columnas binarias.

## P

- **Pandera** — Librería de validación de _DataFrames_ basada en esquemas declarativos.
- **PII** _(Personally Identifiable Information)_ — Información personal identificable. No se utiliza en este proyecto.
- **PR-AUC** _(Precision-Recall AUC)_ — AUC sobre la curva precision-recall; más robusto que ROC-AUC ante desbalance.
- **PSI** _(Population Stability Index)_ — Índice que mide el desplazamiento entre dos distribuciones; umbral típico < 0.10.
- **Pydantic** — Librería de Python para validación de datos basada en _type hints_.

## R

- **Recall** — Sensibilidad; proporción de positivos reales correctamente identificados.
- **Retención** — Tasa complementaria a _churn_: % de clientes activos al inicio del período que permanecen activos al final.
- **ROC-AUC** — Área bajo la curva ROC; mide la capacidad de _ranking_ del modelo.
- **ROI** _(Return on Investment)_ — Retorno sobre la inversión; en este proyecto se calcula como `(MRR retenido − costo de campaña) / costo de campaña`.

## S

- **SaaS** _(Software as a Service)_ — Modelo de distribución de software basado en suscripción.
- **SHAP** _(SHapley Additive exPlanations)_ — Marco de explicabilidad basado en valores de Shapley.
- **SMART** — Especificación de objetivos: _Specific, Measurable, Achievable, Relevant, Time-bound_.
- **Structlog** — Librería de _logging_ estructurado para Python.

## T

- **TDSP** _(Team Data Science Process)_ — Metodología de Microsoft adoptada por el Mindlab UNAL para el diplomado.
- **Tenure** — Tiempo (típicamente en meses) que un cliente ha estado activo desde que contrató el servicio.
- **Top-decile** — Decil superior; conjunto del 10 % de clientes con mayor probabilidad predicha de _churn_.
- **Typer** — Librería de Python para construir interfaces de línea de comandos (CLI) basada en _type hints_.

## U

- **Uplift modeling** — Modelado del efecto causal de una intervención sobre un individuo. Fuera del alcance de la versión inicial.
- **UNAL** — Universidad Nacional de Colombia.

## X

- **XGBoost** — Implementación clásica de _gradient boosting_ ampliamente usada en competencias y en industria.
