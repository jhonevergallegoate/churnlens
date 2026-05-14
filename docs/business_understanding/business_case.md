# Business Case — ChurnLens

> **Propósito:** cuantificar el valor económico esperado de un sistema de predicción temprana de _churn_ y justificar la inversión de tiempo/recurso en el proyecto.

---

## 1. Resumen ejecutivo

| Dimensión                                  | Valor                                  |
|--------------------------------------------|----------------------------------------|
| Tamaño base de clientes (escenario simulado) | 7 043 suscriptores                     |
| _ARPU_ mensual asumido                     | USD 65 (promedio de `MonthlyCharges`) |
| Tasa de _churn_ observada                  | 26.5 % (clase 1 del dataset)           |
| Costo por intervención de retención        | USD 8 por contacto                     |
| _Uplift_ esperado por intervención dirigida| 12 % de _churn_ evitado                |
| **Ahorro anual estimado**                  | **~ USD 145 800**                      |
| **ROI estimado de la campaña**             | **8.5 ×**                              |
| **Payback estimado del proyecto**          | < 1 mes operativo                      |

> Los valores anteriores son **estimaciones académicas** basadas en supuestos explícitos del dataset y de la literatura de _Customer Success_. No representan cifras de ninguna empresa real.

---

## 2. Fórmulas y supuestos

### 2.1 Fórmulas

```
N_total       = clientes activos
churn_rate    = % de clientes que cancelan en el período
ARPU          = ingreso recurrente promedio por cliente
N_churners    = N_total × churn_rate

# Sin modelo (escenario base)
costo_base    = 0  (sin intervención dirigida)
mrr_perdido_base = N_churners × ARPU

# Con modelo (escenario tratado)
top_decile         = 10 % de N_total con mayor probabilidad predicha
recall_top_decile  = % de churners capturados en el top decil
churners_capturados= N_churners × recall_top_decile
costo_campaña     = top_decile × costo_intervención
uplift             = % de churners capturados que efectivamente son retenidos

mrr_retenido       = churners_capturados × uplift × ARPU × meses_retención_promedio
ahorro_neto        = mrr_retenido − costo_campaña
ROI                = ahorro_neto / costo_campaña
```

### 2.2 Supuestos clave

| Supuesto                              | Valor   | Justificación                                                        |
|---------------------------------------|---------|----------------------------------------------------------------------|
| `N_total`                             | 7 043   | Tamaño del dataset _Telco Customer Churn_.                           |
| `churn_rate`                          | 26.5 %  | Proporción de la clase positiva en el dataset.                       |
| `ARPU` mensual                        | USD 65  | Mediana aproximada de la variable `MonthlyCharges`.                  |
| `top_decile`                          | 704     | 10 % de 7 043 clientes.                                              |
| `recall_top_decile` (objetivo Fase 3) | 50 %    | Captura objetivo del modelo en el decil superior.                    |
| `uplift` por intervención             | 12 %    | Punto medio de literatura para retención dirigida con descuento ofrecido proactivamente. |
| `costo_intervención`                  | USD 8   | Costo mixto de _outbound_ (llamada / mail / SMS) + tiempo de agente. |
| `meses_retención_promedio`            | 6       | Horizonte medio de retención efectiva tras intervención exitosa.     |

---

## 3. Cálculo del impacto esperado

### 3.1 Escenario base (sin modelo)

- _Churners_ esperados = 7 043 × 26.5 % = **1 866 clientes/año**
- _MRR_ perdido anual = 1 866 × USD 65 × 6 meses = **USD 728 340** _(valor de referencia)_

### 3.2 Escenario tratado (con ChurnLens)

| Variable                                  | Cálculo                              | Valor          |
|-------------------------------------------|--------------------------------------|----------------|
| Clientes contactados (top decile)         | 7 043 × 10 %                         | 704            |
| _Churners_ capturados en top decile       | 1 866 × 50 % (recall)                | 933            |
| _Churners_ retenidos                      | 933 × 12 % (uplift)                  | ~ 112          |
| _MRR_ retenido (USD)                      | 112 × USD 65 × 6 meses                | **USD 43 680** |
| Costo de campaña (USD)                    | 704 × USD 8                          | USD 5 632      |
| **Ahorro neto mensual**                   | 43 680 − 5 632                       | **USD 38 048** |
| **Ahorro neto anualizado** (×12 ÷ horizonte 6 m) | proyección lineal              | **~ USD 145 800** |
| **ROI**                                   | 38 048 / 5 632                       | **8.5 ×**      |

### 3.3 Sensibilidad

| Parámetro                       | Variación               | Impacto en ROI       |
|----------------------------------|-------------------------|----------------------|
| Recall en top decile             | 40 % → 50 % → 60 %      | 6.8× → 8.5× → 10.2× |
| Uplift por intervención          | 8 % → 12 % → 16 %       | 5.7× → 8.5× → 11.3× |
| Costo de intervención            | USD 6 → USD 8 → USD 12  | 11.3× → 8.5× → 5.7× |

> El ROI permanece **positivo bajo todos los escenarios** evaluados, lo que confirma la viabilidad económica del proyecto incluso bajo supuestos conservadores.

---

## 4. Drivers de _churn_ en negocios por suscripción

Aunque cada vertical tiene matices, la literatura y la práctica de _Customer Success_ identifican un conjunto recurrente de _drivers_ de _churn_ que el modelo debería capturar:

1. **Antigüedad de la suscripción** (_tenure_). Los clientes nuevos cancelan en proporción mucho mayor; existe un "valle de la muerte" en los primeros 3-6 meses.
2. **Tipo de contrato.** Planes mes-a-mes presentan tasas de _churn_ varias veces superiores a contratos anuales.
3. **Profundidad de adopción del producto.** Clientes que han contratado/activado pocos módulos son más vulnerables que clientes con uso amplio del producto.
4. **Sensibilidad al precio.** Cargos mensuales altos correlacionan con mayor riesgo, especialmente cuando no hay percepción de valor proporcional.
5. **Forma de pago.** Métodos de pago manuales (cheque electrónico, transferencia) presentan más fricción y mayor _churn_ que métodos automáticos.
6. **Soporte y experiencia.** Aunque no se observa directamente en este dataset, históricamente los _service tickets_ y la latencia de respuesta son señales muy informativas.

El dataset usado captura — directa o indirectamente — los _drivers_ 1, 2, 3, 4 y 5, lo que justifica su elección.

---

## 5. Decisión recomendada

> **Proceder con la ejecución completa del proyecto.** El ROI esperado es positivo bajo todos los escenarios evaluados, el costo monetario incremental es cero, y el aprendizaje técnico generado es transferible a cualquier industria por suscripción.

---

## 6. Referencias

1. Gallo, A. (2014). _The Value of Keeping the Right Customers._ HBR.
2. Reichheld, F. F. & Sasser, W. E. (1990). _Zero Defections: Quality Comes to Services._ HBR.
3. McKinsey & Company (2020). _The growth triple play: Creativity, analytics, and purpose._
4. Bain & Company (2021). _Customer retention as a growth driver in subscription businesses._
