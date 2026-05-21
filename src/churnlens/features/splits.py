"""Particionado estratificado del dataset.

Convención del proyecto:

* **70 %** _train_.
* **15 %** _validation_ (selección de hiperparámetros, _early stopping_).
* **15 %** _test_ (held-out, se toca **solo al cierre** de la Fase 4).

La estratificación se hace sobre la variable objetivo binarizada para
preservar la tasa de churn (~26.5 %) en los tres conjuntos.

Se usa una semilla fija (`settings.random_seed`, default 42) para que
las particiones sean **bit-exactas** entre ejecuciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

from churnlens.features.preprocessing import TARGET_COL, binarize_target

_TEST_FRACTION: Final[float] = 0.15
_VAL_FRACTION: Final[float] = 0.15


@dataclass(frozen=True)
class SplitResult:
    """Contenedor inmutable con los tres conjuntos resultantes del split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        """Devuelve `{split: (n_filas, n_cols)}` — útil para logging."""
        return {
            "train": self.X_train.shape,
            "val": self.X_val.shape,
            "test": self.X_test.shape,
        }

    @property
    def target_rates(self) -> dict[str, float]:
        """Tasa de positivos en cada split — debe ser ~igual entre splits."""
        return {
            "train": float(self.y_train.mean()),
            "val": float(self.y_val.mean()),
            "test": float(self.y_test.mean()),
        }


def stratified_split(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    test_size: float = _TEST_FRACTION,
    val_size: float = _VAL_FRACTION,
    random_state: int = 42,
) -> SplitResult:
    """Particiona el DataFrame en train / val / test estratificado.

    Args:
        df: DataFrame que contiene tanto las _features_ como la columna
            objetivo (`Yes`/`No` o ya binarizada).
        target_col: nombre de la columna objetivo.
        test_size: proporción del conjunto held-out.
        val_size: proporción del conjunto de validación.
        random_state: semilla para reproducibilidad.

    Returns:
        :class:`SplitResult` con los seis arrays / series listos para uso.

    Raises:
        ValueError: si las proporciones no son válidas o falta la columna
            objetivo.
    """
    if target_col not in df.columns:
        msg = f"Falta la columna objetivo '{target_col}' en el DataFrame."
        raise ValueError(msg)
    if not 0 < test_size < 1 or not 0 < val_size < 1 or (test_size + val_size) >= 1:
        msg = (
            "test_size y val_size deben ser fracciones positivas cuya suma sea "
            f"menor que 1. Recibido: test={test_size}, val={val_size}."
        )
        raise ValueError(msg)

    y_raw = df[target_col]
    y = y_raw if pd.api.types.is_integer_dtype(y_raw) else binarize_target(y_raw)
    x = df.drop(columns=[target_col])

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Reescala val_size para que sea proporción del remanente (train+val).
    val_size_rescaled = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_size_rescaled,
        stratify=y_trainval,
        random_state=random_state,
    )

    return SplitResult(
        X_train=x_train.reset_index(drop=True),
        X_val=x_val.reset_index(drop=True),
        X_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
