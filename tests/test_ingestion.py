"""
Pruebas Unitarias para el Módulo de Ingesta y Validación de Datos.
Verifica carga de datos, validación sintáctica de cabeceras mediante regex (módulo re)
y control de consistencia fisiológica en registros clínicos.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data_ingestion import HeartFailureDataPipeline


@pytest.fixture
def sample_valid_dataframe() -> pd.DataFrame:
    """Fixture que provee un DataFrame sintético con esquema clínico válido."""
    return pd.DataFrame({
        "age": [55.0, 65.0, 70.0],
        "anaemia": [0, 1, 0],
        "creatinine_phosphokinase": [150.0, 250.0, 300.0],
        "diabetes": [0, 0, 1],
        "ejection_fraction": [35.0, 40.0, 25.0],
        "high_blood_pressure": [1, 0, 1],
        "platelets": [250000.0, 300000.0, 180000.0],
        "serum_creatinine": [1.1, 1.4, 2.2],
        "serum_sodium": [136.0, 140.0, 132.0],
        "sex": [1, 0, 1],
        "smoking": [0, 1, 0],
        "time": [100.0, 150.0, 30.0],
        "DEATH_EVENT": [0, 0, 1],
    })


def test_schema_validation_success(sample_valid_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    """Valida que un dataset con nombres conformes a regex y columnas requeridas sea aceptado."""
    raw_csv = tmp_path / "raw.csv"
    processed_csv = tmp_path / "proc.csv"
    sample_valid_dataframe.to_csv(raw_csv, index=False)

    pipeline = HeartFailureDataPipeline(raw_path=raw_csv, processed_path=processed_csv)
    df = pipeline.load_data()
    assert pipeline.validate_schema(df) is True


def test_schema_validation_regex_failure(sample_valid_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    """Valida que cabeceras con caracteres ilegales (violando regex) arrojen ValueError."""
    bad_df = sample_valid_dataframe.copy()
    bad_df.rename(columns={"age": "edad#invalida@"}, inplace=True)
    raw_csv = tmp_path / "bad_raw.csv"
    processed_csv = tmp_path / "proc.csv"
    bad_df.to_csv(raw_csv, index=False)

    pipeline = HeartFailureDataPipeline(raw_path=raw_csv, processed_path=processed_csv)
    df = pipeline.load_data()
    with pytest.raises(ValueError, match="viola el formato alfanumérico"):
        pipeline.validate_schema(df)


def test_missing_target_column(sample_valid_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    """Valida que la ausencia de la columna objetivo DEATH_EVENT genere una excepción controlada."""
    bad_df = sample_valid_dataframe.drop(columns=["DEATH_EVENT"])
    raw_csv = tmp_path / "no_target.csv"
    processed_csv = tmp_path / "proc.csv"
    bad_df.to_csv(raw_csv, index=False)

    pipeline = HeartFailureDataPipeline(raw_path=raw_csv, processed_path=processed_csv)
    df = pipeline.load_data()
    with pytest.raises(ValueError, match="Variable objetivo requerida"):
        pipeline.validate_schema(df)


def test_clean_and_verify_duplicates_removal(sample_valid_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    """Verifica que el método clean_and_verify identifique y descarte duplicados."""
    duplicated_df = pd.concat([sample_valid_dataframe, sample_valid_dataframe.iloc[[0]]], ignore_index=True)
    raw_csv = tmp_path / "dups.csv"
    processed_csv = tmp_path / "proc.csv"
    duplicated_df.to_csv(raw_csv, index=False)

    pipeline = HeartFailureDataPipeline(raw_path=raw_csv, processed_path=processed_csv)
    cleaned = pipeline.clean_and_verify(duplicated_df)
    assert len(cleaned) == len(sample_valid_dataframe)
