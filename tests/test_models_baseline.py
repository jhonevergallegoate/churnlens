"""Tests para `churnlens.models.baseline`."""

from __future__ import annotations

import numpy as np

from churnlens.models.baseline import (
    BASELINE_MODEL_NAMES,
    build_baseline_estimators,
)


def test_build_baseline_estimators_returns_expected_names() -> None:
    estimators = build_baseline_estimators(random_state=0)
    assert set(estimators) == set(BASELINE_MODEL_NAMES)
    assert len(estimators) == 4


def test_baselines_fit_and_predict_proba() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    y = (rng.uniform(size=80) > 0.7).astype("int8")
    for name, estimator in build_baseline_estimators(random_state=0).items():
        estimator.fit(x, y)
        proba = estimator.predict_proba(x)
        assert proba.shape == (80, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), name
