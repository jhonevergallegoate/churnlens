# Privacidad y cumplimiento — ChurnLens

> **Propósito:** documentar el tratamiento de datos personales del proyecto y su alineación con las normativas aplicables. Aunque el proyecto utiliza exclusivamente datos públicos sin PII, se redacta este documento como ejercicio metodológico y para acercarse al estándar que exigiría un entorno productivo real.

---

## 1. Categorización de los datos utilizados

| Categoría                                  | ¿Presente en el proyecto? | Comentario                                                                            |
|--------------------------------------------|---------------------------|----------------------------------------------------------------------------------------|
| Datos personales identificables (PII)      | **No**                    | El `customerID` es sintético (formato `^\d{4}-[A-Z]{5}$`) y no permite reidentificación. |
| Datos personales sensibles                 | **No** (en sentido estricto) | `gender` y `SeniorCitizen` son atributos sensibles para análisis de _fairness_, pero no son PII por sí mismos. |
| Datos financieros                          | **Indirecto**             | `MonthlyCharges` y `TotalCharges` reflejan facturación pero no asocian a una persona real. |
| Datos de salud                             | **No**                    | —                                                                                      |
| Datos biométricos                          | **No**                    | —                                                                                      |
| Datos de localización                      | **No**                    | El dataset no incluye geolocalización ni códigos postales.                              |
| Datos de menores                           | **No**                    | El dataset es de clientes adultos titulares de un servicio.                             |

---

## 2. Normativas relevantes consideradas

| Normativa                                                            | Aplicabilidad al proyecto                                                                  |
|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **Ley 1581 de 2012 + Decreto 1377 de 2013** (Habeas Data — Colombia)  | Sin obligaciones activas (no hay datos personales). Se respetan los principios como ejercicio. |
| **GDPR / Reglamento (UE) 2016/679**                                  | Sin obligaciones activas. Los principios de minimización y propósito se respetan.           |
| **AI Act — Reglamento (UE) 2024/1689**                               | El sistema de _scoring_ de churn no cae en categorías de "alto riesgo"; se documenta como referencia. |
| **ISO/IEC 27001**                                                    | Las prácticas de control de versiones, hashes y `.env` no versionado se alinean.            |
| **OWASP Top 10**                                                     | Sin exposición pública en la Fase 1.                                                       |

---

## 3. Principios aplicados

| Principio                          | Cómo se aplica                                                                                                                            |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **Minimización**                   | Solo se cargan las 21 variables del dataset; no se enriquece con fuentes adicionales sin un análisis previo.                              |
| **Finalidad**                      | Los datos se usan exclusivamente para construir un modelo predictivo académico.                                                            |
| **Calidad**                         | Validación obligatoria por Pandera + hashes sobre el _raw_.                                                                                |
| **Seguridad técnica**              | `.env` ignorado por git; sin credenciales hardcoded; CI con _secret scanning_ posible.                                                     |
| **Transparencia**                  | Toda la documentación es pública en el repositorio; los _changelogs_ están en el historial de commits.                                     |
| **Limitación de retención**        | Los datos crudos viven solo en el sistema de archivos local del operador; no se persisten en bases compartidas.                            |
| **Atribución**                      | El dataset se atribuye correctamente a IBM en `LICENSE` y `data_definition.md`.                                                            |

---

## 4. Prácticas de seguridad técnica del repositorio

| Práctica                                       | Implementación                                                  |
|------------------------------------------------|-----------------------------------------------------------------|
| Variables sensibles fuera del control de versiones | `.gitignore` excluye `.env`, `secrets.yaml`, `*.pem`, `*.key`.   |
| Detección de _secrets_ accidentales            | Hook `detect-private-key` de pre-commit.                        |
| Detección de archivos grandes                   | Hook `check-added-large-files` con cap de 2 MB.                  |
| Reproducibilidad de dependencias                | `pyproject.toml` con _ranges_ acotados.                          |
| Integridad de datos                            | MD5 + SHA-256 sobre el _raw_ persistidos en `.checksums.json`.    |
| CI con linting estricto                        | `ruff`, `mypy --strict`, `pytest --cov` en cada PR.              |

---

## 5. ¿Qué cambiaría si esto fuera productivo real?

En un escenario productivo con datos reales de clientes, se sumarían los siguientes controles:

1. **Anonimización / seudonimización** de identificadores antes de cargar al modelo.
2. **Tratamiento documentado** con base legal explícita (consentimiento o interés legítimo).
3. **Auditoría DPIA** (_Data Protection Impact Assessment_) para cualquier feature de _scoring_.
4. **Encriptación at-rest + in-transit** del _raw_ y del modelo.
5. **Retención limitada** con políticas automatizadas de borrado.
6. **Trazabilidad por consentimiento** — cada cliente debe poder solicitar el opt-out del scoring.
7. **Acceso de mínimo privilegio** al modelo y los logs.
8. **Plan de respuesta a incidentes** documentado (DPO + notificación a autoridad < 72 h en caso de _breach_).

Estas prácticas se nombran aquí como referencia para una continuación post-académica del proyecto.

---

## 6. Procedimiento de _data subject rights_ (referencia)

> *Aplicable si el proyecto evolucionara a un escenario productivo real con clientes reales.*

| Derecho               | Mecanismo                                                                          |
|-----------------------|------------------------------------------------------------------------------------|
| Acceso                | API interna `GET /me/score-explanation`.                                            |
| Rectificación         | Si el cliente reporta datos incorrectos, se actualizan en la fuente y se reentrena. |
| Supresión / olvido    | El cliente puede solicitar exclusión del set de entrenamiento del próximo ciclo.    |
| Portabilidad          | Export en JSON estructurado.                                                       |
| Oposición al _scoring_| Opt-out total; el cliente queda excluido de campañas dirigidas por el modelo.       |
| Decisión automatizada | Garantía de revisión humana en todas las acciones materiales.                       |

---

## 7. Referencias

1. República de Colombia. _Ley Estatutaria 1581 de 2012_.
2. Unión Europea. _Reglamento (UE) 2016/679_ (GDPR).
3. Unión Europea. _Reglamento (UE) 2024/1689_ (AI Act).
4. NIST AI Risk Management Framework (2023).
5. ISO/IEC 27001:2022 — Information Security Management Systems.
