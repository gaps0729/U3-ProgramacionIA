"""
Módulo de Ingesta y Validación de Datos Clínicos.
Implementa clases abstractas e interfaces bajo POO, validación rigurosa de esquema
con expresiones regulares (módulo re) y trazabilidad estructurada con logging.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import (
    ClinicalSchemaConfig,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    setup_logger,
)


class BaseDataPipeline(ABC):
    """
    Clase base abstracta que define el contrato de interfaz para pipelines
    de ingesta y validación de datos en sistemas de Inteligencia Artificial.
    """

    def __init__(self, raw_path: Path, processed_path: Path) -> None:
        """
        Inicializa el pipeline de datos base.

        Args:
            raw_path: Ruta al archivo fuente de datos en crudo.
            processed_path: Ruta de destino para el dataset verificado y limpio.
        """
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Carga el conjunto de datos desde el almacenamiento persistente."""
        pass

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Valida que el DataFrame cumpla con las especificaciones de esquema y tipos."""
        pass

    @abstractmethod
    def clean_and_verify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica reglas de sanitización, detección de anomalías y consistencia de dominio."""
        pass


class HeartFailureDataPipeline(BaseDataPipeline):
    """
    Implementación concreta para la ingesta, validación sintáctica mediante regex
    y verificación fisiológica del dataset de insuficiencia cardíaca.
    """

    def __init__(
        self,
        raw_path: Path = RAW_DATA_PATH,
        processed_path: Path = PROCESSED_DATA_PATH,
        schema_config: Optional[ClinicalSchemaConfig] = None,
    ) -> None:
        """
        Inicializa el pipeline de datos para insuficiencia cardíaca.

        Args:
            raw_path: Ruta al archivo CSV con los datos crudos.
            processed_path: Ruta para almacenar el CSV procesado.
            schema_config: Configuración de esquema clínico. Si es None, usa la predeterminada.
        """
        super().__init__(raw_path, processed_path)
        self.schema = schema_config or ClinicalSchemaConfig()

    def load_data(self) -> pd.DataFrame:
        """
        Lee el archivo CSV desde disco validando existencia y codificación.

        Returns:
            pd.DataFrame con los datos crudos cargados.

        Raises:
            FileNotFoundError: Si el archivo fuente no existe.
            ValueError: Si el archivo cargado está vacío.
        """
        self.logger.info("Iniciando carga de datos desde: %s", self.raw_path)
        if not self.raw_path.exists():
            error_msg = f"El archivo fuente no fue encontrado en la ruta: {self.raw_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            df = pd.read_csv(self.raw_path)
            self.logger.info("Datos cargados exitosamente. Dimensiones: %s filas, %s columnas", *df.shape)
            if df.empty:
                raise ValueError("El archivo CSV cargado se encuentra vacío.")
            return df
        except Exception as exc:
            self.logger.exception("Error crítico durante la lectura del CSV: %s", exc)
            raise

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Verifica la integridad de las columnas usando expresiones regulares (re)
        y comprueba la presencia de todas las variables clínicas requeridas.

        Args:
            df: DataFrame a auditar.

        Returns:
            True si el esquema es 100% conforme.

        Raises:
            ValueError: Si faltan columnas requeridas o alguna cabecera viola la sintaxis regex.
        """
        self.logger.info("Iniciando auditoría de esquema y sintaxis de cabeceras mediante regex...")

        # 1. Validación de sintaxis en nombres de columnas usando regex
        for col in df.columns:
            if not self.schema.valid_column_name_regex.match(str(col)):
                error_msg = f"La columna '{col}' viola el formato alfanumérico estándar permitido."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # 2. Comprobación de presencia de variable objetivo
        if self.schema.target_column not in df.columns:
            error_msg = f"Variable objetivo requerida '{self.schema.target_column}' ausente en el dataset."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 3. Comprobación de variables predictoras
        expected_features = set(self.schema.numerical_features + self.schema.binary_features)
        missing_features = expected_features - set(df.columns)
        if missing_features:
            error_msg = f"Faltan las siguientes variables clínicas obligatorias: {missing_features}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(
            "Auditoría de esquema superada exitosamente. Total variables verificadas: %d",
            len(expected_features) + 1,
        )
        return True

    def clean_and_verify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica reglas de negocio clínico:
        - Verificación de duplicados y valores nulos.
        - Auditoría de rangos fisiológicos (emite WARNING si hay valores sospechosos).
        - Conversión explícita de tipos de datos.

        Args:
            df: DataFrame con los datos verificados sintácticamente.

        Returns:
            pd.DataFrame depurado y consistente.
        """
        self.logger.info("Iniciando limpieza y verificación fisiológica de variables...")
        cleaned_df = df.copy()

        # Detección y tratamiento de duplicados
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        dropped_duplicates = initial_rows - len(cleaned_df)
        if dropped_duplicates > 0:
            self.logger.warning("Se detectaron y eliminaron %d registros duplicados.", dropped_duplicates)

        # Detección de valores faltantes (NaN)
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            self.logger.warning(
                "Se encontraron %d valores faltantes. Se aplicará imputación en preprocesamiento.",
                missing_count,
            )
        else:
            self.logger.info("Conjunto de datos libre de valores nulos (100% completo).")

        # Verificación de rangos plausibles en variables numéricas
        for feature, (min_val, max_val) in self.schema.plausible_ranges.items():
            if feature in cleaned_df.columns:
                out_of_bounds = cleaned_df[(cleaned_df[feature] < min_val) | (cleaned_df[feature] > max_val)]
                if not out_of_bounds.empty:
                    self.logger.warning(
                        "Alerta fisiológica: %d registros en '%s' fuera del rango habitual [%s, %s].",
                        len(out_of_bounds),
                        feature,
                        min_val,
                        max_val,
                    )

        # Verificación de consistencia en variables binarias (deben ser estrictamente 0 o 1)
        for bin_feature in self.schema.binary_features:
            invalid_bin = cleaned_df[~cleaned_df[bin_feature].isin([0, 1])]
            if not invalid_bin.empty:
                error_msg = f"La variable binaria '{bin_feature}' contiene valores no válidos distintos de 0 o 1."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Persistir el dataset procesado
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(self.processed_path, index=False)
        self.logger.info("Dataset limpio persistido satisfactoriamente en: %s", self.processed_path)

        return cleaned_df

    def run(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Orquesta el ciclo completo de ingesta y validación de datos.

        Returns:
            Tupla (X, y) donde X es la matriz de predictores e y es el vector objetivo.
        """
        raw_df = self.load_data()
        self.validate_schema(raw_df)
        cleaned_df = self.clean_and_verify(raw_df)

        X = cleaned_df.drop(columns=[self.schema.target_column])
        y = cleaned_df[self.schema.target_column]

        return X, y
