"""
Pruebas Unitarias para el Orquestador PredictivePipeline.
Verifica inferencia individual de pacientes, detección de biomarcadores de alerta
y consistencia de probabilidades predictivas.
"""

import pytest

from src.pipeline import PredictivePipeline


@pytest.fixture
def sample_patient_low_risk() -> dict:
    """Paciente con perfil clínico favorable (bajo riesgo)."""
    return {
        "age": 45.0,
        "anaemia": 0,
        "creatinine_phosphokinase": 120.0,
        "diabetes": 0,
        "ejection_fraction": 55.0,
        "high_blood_pressure": 0,
        "platelets": 280000.0,
        "serum_creatinine": 0.8,
        "serum_sodium": 140.0,
        "sex": 1,
        "smoking": 0,
        "time": 250.0,
    }


@pytest.fixture
def sample_patient_high_risk() -> dict:
    """Paciente con perfil clínico crítico (alto riesgo)."""
    return {
        "age": 78.0,
        "anaemia": 1,
        "creatinine_phosphokinase": 582.0,
        "diabetes": 1,
        "ejection_fraction": 20.0,
        "high_blood_pressure": 1,
        "platelets": 140000.0,
        "serum_creatinine": 3.2,
        "serum_sodium": 130.0,
        "sex": 0,
        "smoking": 1,
        "time": 10.0,
    }


def test_predict_patient_low_vs_high(
    sample_patient_low_risk: dict, sample_patient_high_risk: dict
) -> None:
    """
    Verifica que el modelo distinga cuantitativamente entre un paciente
    de bajo riesgo y un paciente crítico.
    """
    pipeline = PredictivePipeline()
    pipeline.load_pipeline()

    res_low = pipeline.predict_patient(sample_patient_low_risk)
    res_high = pipeline.predict_patient(sample_patient_high_risk)

    assert "probability_mortality" in res_low
    assert "risk_category" in res_low
    assert 0.0 <= res_low["probability_mortality"] <= 1.0
    assert 0.0 <= res_high["probability_mortality"] <= 1.0

    # La probabilidad del paciente crítico debe ser significativamente mayor
    assert res_high["probability_mortality"] > res_low["probability_mortality"]
    assert len(res_high["alert_factors"]) > 0

