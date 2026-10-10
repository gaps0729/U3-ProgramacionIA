"""
Módulo de Preprocesamiento y Feature Engineering Clínico.
Implementa clases abstractas y transformadores de variables siguiendo
el paradigma de Programación Orientada a Objetos (POO), garantizando la
ausencia de fuga de datos (data leakage) y trazabilidad completa con logging.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import ClinicalSchemaConfig, setup_logger


class BaseFeatureEngineer(ABC):
    """
    Clase abstracta que define el contrato de transformación y feature engineering
    para pipelines de machine learning.
    """

    def __init__(self) -> None:
        """Inicializa el transformador base con logger jerárquico."""
        self.logger = setup_logger(self.__class__.__name__)
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseFeatureEngineer":
        """Ajusta los parámetros del transformador sobre el conjunto de entrenamiento."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Aplica la transformación sobre nuevos conjuntos de datos sin alterar parámetros."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Ajusta y transforma secuencialmente sobre el conjunto de entrenamiento.

        Args:
            X: Matriz de características.
            y: Vector objetivo opcional.

        Returns:
            np.ndarray con las características transformadas.
        """
        return self.fit(X, y).transform(X)


class ClinicalFeatureEngineer(BaseFeatureEngineer):
    """
    Transformador de variables biomédicas adaptado al perfil clínico de insuficiencia cardíaca.
    Aplica imputación robusta y estandarización z-score sobre variables continuas,
    preservando la integridad de variables binarias.
    """

    def __init__(self, schema_config: Optional[ClinicalSchemaConfig] = None) -> None:
        """
        Inicializa el transformador clínico.

        Args:
            schema_config: Configuración del esquema de variables.
        """
        super().__init__()
        self.schema = schema_config or ClinicalSchemaConfig()
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str] = []

    def _build_column_transformer(self) -> ColumnTransformer:
        """
        Construye el transformador compuesto por columnas.
        Aplica imputación por mediana + estandarización para variables numéricas,
        e imputación por moda para variables categóricas binarias.

        Returns:
            ColumnTransformer configurado.
        """
        from sklearn.pipeline import Pipeline

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        bin_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])

        composite_transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.schema.numerical_features),
                ("bin", bin_pipeline, self.schema.binary_features),
            ],
            remainder="drop",
        )
        return composite_transformer

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClinicalFeatureEngineer":
        """
        Calcula medias, varianzas y estadísticas del conjunto de datos de entrenamiento.

        Args:
            X: DataFrame con las variables predictoras.
            y: Vector objetivo opcional (no utilizado en transformaciones no supervisadas).

        Returns:
            Instancia ajustada de ClinicalFeatureEngineer.

        Raises:
            ValueError: Si X no contiene las columnas requeridas por el esquema.
        """
        self.logger.info("Iniciando ajuste (fit) del transformador sobre %d observaciones...", len(X))

        missing_cols = set(self.schema.numerical_features + self.schema.binary_features) - set(X.columns)
        if missing_cols:
            error_msg = f"No se puede ajustar el preprocesador. Faltan columnas: {missing_cols}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.pipeline = self._build_column_transformer()
        self.pipeline.fit(X)
        self.feature_names_out = self.schema.numerical_features + self.schema.binary_features
        self.is_fitted = True

        self.logger.info(
            "Transformador clínico ajustado exitosamente. Variables procesadas: %d (%d numéricas, %d binarias)",
            len(self.feature_names_out),
            len(self.schema.numerical_features),
            len(self.schema.binary_features),
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Aplica la transformación aprendida a un conjunto de datos (test o inferencia en tiempo real).

        Args:
            X: DataFrame a transformar.

        Returns:
            np.ndarray de características normalizadas.

        Raises:
            RuntimeError: Si se invoca transform antes de ajustar (fit) el objeto.
        """
        if not self.is_fitted or self.pipeline is None:
            error_msg = "El transformador debe ser ajustado (fit) antes de llamar a transform."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.debug("Transformando lote de datos clínicos con dimensiones %s", X.shape)
        transformed_array = self.pipeline.transform(X)
        return transformed_array

    def get_feature_names(self) -> List[str]:
        """
        Devuelve los nombres ordenados de las características procesadas.

        Returns:
            Lista de cadenas con los nombres de características.
        """
        return self.feature_names_out
