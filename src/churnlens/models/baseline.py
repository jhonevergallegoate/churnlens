"""Modelos de línea base (Fase 3).

Define las **referencias mínimas** contra las cuales se compara cualquier
modelo "real" del proyecto:

* ``dummy_stratified`` — predice respetando la distribución a priori.
* ``dummy_most_frequent`` — predice siempre la clase mayoritaria.
* ``dummy_prior`` — predice la probabilidad a priori para todos.
* ``logreg_balanced`` — Logistic Regression L2 con ``class_weight='balanced'``.

Cualquier modelo del proyecto debe **superar de forma material** a estos
baselines en PR-AUC y F1 (en ese orden de prioridad). Si no, se considera
que aporta complejidad sin valor.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

BASELINE_MODEL_NAMES: tuple[str, ...] = (
    "dummy_stratified",
    "dummy_most_frequent",
    "dummy_prior",
    "logreg_balanced",
)


def build_baseline_estimators(*, random_state: int = 42) -> dict[str, Any]:
    """Devuelve un dict ``{name: estimator}`` con los baselines del proyecto.

    Args:
        random_state: semilla para los dummies aleatorios y el solver de
            logistic regression.

    Returns:
        Diccionario con cuatro estimadores listos para `fit`.
    """
    return {
        "dummy_stratified": DummyClassifier(
            strategy="stratified",
            random_state=random_state,
        ),
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logreg_balanced": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
        ),
    }
