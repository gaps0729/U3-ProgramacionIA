"""
Módulo de Configuración y Parámetros Globales.
Define rutas, constantes, esquemas clínicos, límites fisiológicos y configuración
estructurada de logging para todo el pipeline de IA.
"""

from dataclasses import dataclass, field
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
from typing import Dict, List, Tuple


# ==============================================================================
# Rutas del Proyecto
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "heart_failure_clinical_records_dataset.csv"
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed" / "cleaned_heart_records.csv"
MODELS_DIR: Path = BASE_DIR / "models"
MODEL_ARTIFACT_PATH: Path = MODELS_DIR / "best_heart_failure_model.joblib"
METRICS_PATH: Path = MODELS_DIR / "evaluation_metrics.json"
LOGS_DIR: Path = BASE_DIR / "logs"
LOG_FILE_PATH: Path = LOGS_DIR / "pipeline.log"


# ==============================================================================
# Definición de Esquema de Datos y Reglas de Dominio Clínico
# ==============================================================================
@dataclass(frozen=True)
class ClinicalSchemaConfig:
    """Configuración inmutable del esquema de datos clínicos."""

    target_column: str = "DEATH_EVENT"

    numerical_features: List[str] = field(
        default_factory=lambda: [
            "age",
            "creatinine_phosphokinase",
            "ejection_fraction",
            "platelets",
            "serum_creatinine",
            "serum_sodium",
            "time",
        ]
    )

    binary_features: List[str] = field(
        default_factory=lambda: [
            "anaemia",
            "diabetes",
            "high_blood_pressure",
            "sex",
            "smoking",
        ]
    )

    # Rangos fisiológicos plausibles para control de calidad médica
    plausible_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "age": (18.0, 120.0),
            "creatinine_phosphokinase": (10.0, 10000.0),
            "ejection_fraction": (5.0, 90.0),
            "platelets": (20000.0, 1000000.0),
            "serum_creatinine": (0.2, 15.0),
            "serum_sodium": (100.0, 160.0),
            "time": (1.0, 365.0),
        }
    )

    # Patrón de Expresión Regular (módulo re) para validar identificadores clínicos
    patient_id_regex: re.Pattern = field(
        default_factory=lambda: re.compile(r"^PAC-[A-Z]{3}-\d{4,6}$")
    )

    # Patrón de Expresión Regular para verificar cabeceras válidas de columnas
    valid_column_name_regex: re.Pattern = field(
        default_factory=lambda: re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    )


@dataclass
class ModelExperimentConfig:
    """Configuración para particionado, validación cruzada y optimización."""

    test_size: float = 0.2
    random_state: int = 42
    n_splits_cv: int = 5
    scoring_metric: str = "roc_auc"


# ==============================================================================
# Configuración Centralizada de Logging Profesional
# ==============================================================================
def setup_logger(
    name: str = "HeartFailureAIPipeline",
    log_file: Path = LOG_FILE_PATH,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configura y devuelve un logger estructurado con manejadores hacia consola
    y archivo rotativo, garantizando trazabilidad y observabilidad de grado de producción.

    Args:
        name: Identificador del logger jerárquico.
        log_file: Ruta al archivo físico de logs.
        level: Nivel mínimo de registro (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Instancia de logging.Logger configurada.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Manejador a consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Manejador rotativo a archivo (máx 5MB por archivo, conserva 3 backups)
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

