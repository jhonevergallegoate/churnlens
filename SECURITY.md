# Política de seguridad — ChurnLens

## Alcance

Este repositorio aloja un proyecto académico desarrollado en el marco del Diplomado MLDS de la Universidad Nacional de Colombia. **No** procesa datos personales reales, **no** está desplegado en producción y **no** recibe tráfico de usuarios finales.

Aun así, se aplican prácticas estándar de _open-source security_ para mantener el repositorio limpio y reproducible.

## Versiones soportadas

| Versión        | Soporte            |
|----------------|--------------------|
| `0.1.x` (Fase 1) | ✅ activa         |
| `< 0.1.0`      | ❌ no soportada    |

## Reportar una vulnerabilidad

Si encuentra un problema de seguridad — credenciales filtradas, dependencia con CVE conocida, comportamiento inseguro del _loader_, etc. — siga uno de los siguientes caminos:

1. **Preferido:** abra un _security advisory_ privado en GitHub
   → `https://github.com/jhonevergallegoate/churnlens/security/advisories/new`
2. **Alterno:** envíe un correo a `jhgallegoa21@gmail.com` con el asunto `[SECURITY] ChurnLens — <breve descripción>`.

Por favor **no** abra un _issue_ público para problemas de seguridad antes de que se haya coordinado la divulgación.

### Tiempo de respuesta esperado

| Severidad         | Acuse de recibo | Mitigación inicial |
|-------------------|-----------------|--------------------|
| Crítica / Alta    | < 72 h          | < 2 semanas        |
| Media / Baja      | < 1 semana      | _best effort_      |

## Prácticas internas

- **Secret scanning** y **push protection** habilitados en GitHub.
- **Dependabot security updates** habilitado para Python y GitHub Actions.
- **CI obligatorio** (lint + type-check + tests) en cada push y PR a `main`.
- **Pre-commit** local incluye `detect-private-key` y `check-added-large-files`.
- Variables sensibles viven exclusivamente en `.env` (no versionado).
- Hashes MD5 + SHA-256 del dataset _raw_ persistidos para verificar integridad.

## Lo que NO se considera vulnerabilidad

- Problemas que requieren acceso físico o privilegios de administrador local.
- Comportamiento del modelo entrenado (las decisiones del modelo se documentan en la _model card_ y en `ethics_and_fairness.md`, no son vulnerabilidades de software).
- Reportes sobre el dataset original (propiedad de IBM Corp.) — esos deben dirigirse al publicador original.
