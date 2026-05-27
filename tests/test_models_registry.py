"""Tests para `churnlens.models.registry`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier

from churnlens.models.registry import (
    best_model_by,
    list_models,
    load_model,
    save_model,
)


def _fit_dummy() -> DummyClassifier:
    import numpy as np

    rng = np.random.default_rng(0)
    estimator = DummyClassifier(strategy="stratified", random_state=0)
    estimator.fit(rng.normal(size=(50, 3)), rng.choice([0, 1], size=50))
    return estimator


def test_save_load_roundtrip(tmp_path: Path) -> None:
    estimator = _fit_dummy()
    entry = save_model(
        estimator,
        "dummy_test",
        metadata={"metrics": {"val": {"pr_auc": 0.3}}, "feature_set": ["a", "b", "c"]},
        models_dir=tmp_path,
    )
    assert entry.model_path.exists()
    assert entry.metadata_path.exists()
    assert "hash_model" in entry.metadata
    assert entry.metadata["algorithm"] == "DummyClassifier"

    loaded, metadata = load_model("dummy_test", models_dir=tmp_path)
    assert loaded.__class__.__name__ == "DummyClassifier"
    assert metadata["feature_set"] == ["a", "b", "c"]


def test_save_rejects_invalid_name(tmp_path: Path) -> None:
    estimator = _fit_dummy()
    for bad in ("", "with/slash", ".hidden"):
        with pytest.raises(ValueError, match="inválido"):
            save_model(estimator, bad, models_dir=tmp_path)


def test_load_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model("does_not_exist", models_dir=tmp_path)


def test_list_models_empty_dir(tmp_path: Path) -> None:
    assert list_models(models_dir=tmp_path) == []


def test_best_model_by_returns_winner(tmp_path: Path) -> None:
    estimator = _fit_dummy()
    save_model(
        estimator,
        "a",
        metadata={"metrics": {"val": {"pr_auc": 0.42}}},
        models_dir=tmp_path,
    )
    save_model(
        estimator,
        "b",
        metadata={"metrics": {"val": {"pr_auc": 0.51}}},
        models_dir=tmp_path,
    )
    save_model(
        estimator,
        "c",
        metadata={"metrics": {"val": {"pr_auc": 0.10}}},
        models_dir=tmp_path,
    )
    winner = best_model_by("pr_auc", split="val", models_dir=tmp_path)
    assert winner is not None
    assert winner.name == "b"


def test_best_model_by_returns_none_when_no_metric(tmp_path: Path) -> None:
    estimator = _fit_dummy()
    save_model(estimator, "a", metadata={"metrics": {"val": {}}}, models_dir=tmp_path)
    assert best_model_by("pr_auc", split="val", models_dir=tmp_path) is None


def test_list_models_ignores_malformed_manifest(tmp_path: Path) -> None:
    estimator = _fit_dummy()
    save_model(estimator, "good", metadata={"metrics": {}}, models_dir=tmp_path)
    (tmp_path / "broken.metadata.json").write_text("{not-json", encoding="utf-8")
    entries = list_models(models_dir=tmp_path)
    names = [e.name for e in entries]
    assert "good" in names
    assert "broken" not in names
    # Validar que el manifest existente sigue siendo legible
    assert (
        json.loads((tmp_path / "good.metadata.json").read_text(encoding="utf-8"))["name"] == "good"
    )
