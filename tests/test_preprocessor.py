"""
Pruebas Unitarias para el Módulo de Preprocesamiento y Feature Engineering.
Verifica que las transformaciones numéricas y binarias preserven las dimensiones
esperadas y eviten fugas de información (data leakage).
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import ClinicalFeatureEngineer


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Fixture con variables predictoras sintéticas."""
    return pd.DataFrame({
        "age": [50.0, 60.0, 75.0, 80.0],
        "creatinine_phosphokinase": [100.0, 200.0, 300.0, 400.0],
        "ejection_fraction": [30.0, 40.0, 50.0, 20.0],
        "platelets": [200000.0, 250000.0, 300000.0, 150000.0],
        "serum_creatinine": [1.0, 1.2, 2.1, 0.9],
        "serum_sodium": [135.0, 140.0, 130.0, 138.0],
        "time": [50.0, 100.0, 150.0, 200.0],
        "anaemia": [0, 1, 0, 1],
        "diabetes": [0, 0, 1, 0],
        "high_blood_pressure": [1, 0, 1, 0],
        "sex": [1, 0, 1, 1],
        "smoking": [0, 1, 0, 0],
    })


def test_preprocessor_fit_transform_shape(sample_features_df: pd.DataFrame) -> None:
    """Verifica que el preprocesador mantenga las 12 características."""
    engineer = ClinicalFeatureEngineer()
    transformed = engineer.fit_transform(sample_features_df)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (4, 12)
    assert len(engineer.get_feature_names()) == 12


def test_preprocessor_raises_if_not_fitted(sample_features_df: pd.DataFrame) -> None:
    """Verifica que invocar transform sin previo fit levante un RuntimeError."""
    engineer = ClinicalFeatureEngineer()
    with pytest.raises(RuntimeError, match="debe ser ajustado"):
        engineer.transform(sample_features_df)


def test_preprocessor_handles_missing_values(sample_features_df: pd.DataFrame) -> None:
    """Verifica que el preprocesador impute valores nulos sin errores en tiempo de ejecución."""
    df_with_nans = sample_features_df.copy()
    df_with_nans.loc[0, "serum_creatinine"] = np.nan
    df_with_nans.loc[1, "anaemia"] = np.nan

    engineer = ClinicalFeatureEngineer()
    engineer.fit(sample_features_df)
    transformed = engineer.transform(df_with_nans)

    assert not np.isnan(transformed).any()
