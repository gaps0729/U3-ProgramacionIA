"""
Módulo Orquestador del Pipeline End-to-End de IA.
Integra bajo el patrón Facade las etapas de ingesta, preprocesamiento,
entrenamiento, evaluación analítica y serialización persistente, permitiendo
además la inferencia predictiva y explicabilidad individual en tiempo real.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ClinicalSchemaConfig,
    ModelExperimentConfig,
    MODEL_ARTIFACT_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    setup_logger,
)
from src.data_ingestion import HeartFailureDataPipeline
from src.evaluator import ModelEvaluator
from src.model_trainer import ClassificationModelTrainer
from src.preprocessor import ClinicalFeatureEngineer


class PredictivePipeline:
    """
    Orquestador principal del sistema de inteligencia artificial para pronóstico cardiovascular.
    Implementa el flujo completo de ciclo de vida de MLOps: ingesta, entrenamiento,
    evaluación, persistencia e inferencia en producción.
    """

    def __init__(
        self,
        raw_data_path: Path = RAW_DATA_PATH,
        processed_data_path: Path = PROCESSED_DATA_PATH,
        model_artifact_path: Path = MODEL_ARTIFACT_PATH,
        experiment_config: Optional[ModelExperimentConfig] = None,
        schema_config: Optional[ClinicalSchemaConfig] = None,
    ) -> None:
        """
        Inicializa el orquestador con inyección de dependencias.

        Args:
            raw_data_path: Ruta al archivo de datos fuente.
            processed_data_path: Ruta al archivo de datos preprocesados.
            model_artifact_path: Ruta del artefacto binario (.joblib).
            experiment_config: Parámetros del experimento y validación cruzada.
            schema_config: Especificación del esquema de datos.
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.model_artifact_path = model_artifact_path
        self.exp_config = experiment_config or ModelExperimentConfig()
        self.schema = schema_config or ClinicalSchemaConfig()
        self.logger = setup_logger(self.__class__.__name__)

        # Componentes desacoplados (POO)
        self.data_pipeline = HeartFailureDataPipeline(
            raw_path=self.raw_data_path,
            processed_path=self.processed_data_path,
            schema_config=self.schema,
        )
        self.preprocessor = ClinicalFeatureEngineer(schema_config=self.schema)
        self.trainer = ClassificationModelTrainer(experiment_config=self.exp_config)
        self.evaluator = ModelEvaluator()

        self.fitted_model: Optional[Any] = None
        self.is_trained: bool = False

    def train_and_evaluate(self) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de machine learning:
        1. Ingesta y validación de esquema con regex y reglas de dominio.
        2. Particionado estratificado en conjuntos de entrenamiento y prueba.
        3. Ajuste del transformador de variables (feature engineering).
        4. Benchmarking de modelos y optimización de hiperparámetros.
        5. Evaluación diagnóstica en conjunto de prueba y generación de gráficos.
        6. Serialización del modelo y preprocesador a disco.

        Returns:
            Diccionario con las métricas finales de evaluación en prueba.
        """
        self.logger.info("================ INICIANDO PIPELINE DE ENTRENAMIENTO ================")

        # 1. Ingesta y validación
        X_df, y_series = self.data_pipeline.run()

        # 2. Partición estratificada (mitiga data leakage y sesgo de selección)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y_series,
            test_size=self.exp_config.test_size,
            random_state=self.exp_config.random_state,
            stratify=y_series,
        )
        self.logger.info(
            "Partición estratificada completada: Train=%d registros, Test=%d registros",
            len(X_train_df),
            len(X_test_df),
        )

        # 3. Preprocesamiento (fit_transform solo en train, transform en test)
        X_train_transformed = self.preprocessor.fit_transform(X_train_df)
        X_test_transformed = self.preprocessor.transform(X_test_df)

        # 4. Benchmarking y Optimización
        benchmark_df = self.trainer.benchmark_models(
            X=X_train_transformed, y=y_train.to_numpy()
        )
        self.logger.info("Resumen de Benchmarking:\n%s", benchmark_df.to_string(index=False))

        # Seleccionar el modelo de mejor desempeño (RandomForest o GradientBoosting)
        top_model_name = str(benchmark_df.iloc[0]["Model"])
        self.fitted_model = self.trainer.optimize_hyperparameters(
            X=X_train_transformed, y=y_train.to_numpy(), model_name=top_model_name
        )

        # 5. Evaluación analítica en prueba (Holdout)
        y_test_arr = y_test.to_numpy()
        y_pred = self.fitted_model.predict(X_test_transformed)
        y_proba = self.fitted_model.predict_proba(X_test_transformed)[:, 1]

        metrics = self.evaluator.evaluate_test_set(
            y_true=y_test_arr,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name=top_model_name,
        )

        # Generar gráficos analíticos para el informe y repositorio
        feature_names = self.preprocessor.get_feature_names()
        feature_importances = getattr(self.fitted_model, "feature_importances_", None)

        docs_dir = self.model_artifact_path.parent.parent / "docs"
        self.evaluator.generate_evaluation_plots(
            y_true=y_test_arr,
            y_pred=y_pred,
            y_proba=y_proba,
            feature_names=feature_names,
            feature_importances=feature_importances,
            output_dir=docs_dir,
        )

        # 6. Serialización persistente
        self.save_pipeline()
        self.is_trained = True

        self.logger.info("================ PIPELINE FINALIZADO CON ÉXITO ================")
        return metrics

    def save_pipeline(self) -> None:
        """Serializa el modelo, preprocesador y metadatos en un archivo .joblib."""
        self.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "preprocessor": self.preprocessor,
            "model": self.fitted_model,
            "model_name": self.trainer.best_estimator_name,
            "feature_names": self.preprocessor.get_feature_names(),
        }
        joblib.dump(artifact, self.model_artifact_path)
        self.logger.info("Artefacto del pipeline serializado en: %s", self.model_artifact_path)

    def load_pipeline(self) -> None:
        """Carga el pipeline serializado desde el almacenamiento persistente."""
        if not self.model_artifact_path.exists():
            error_msg = f"El artefacto del modelo no existe en: {self.model_artifact_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        artifact = joblib.load(self.model_artifact_path)
        self.preprocessor = artifact["preprocessor"]
        self.fitted_model = artifact["model"]
        self.is_trained = True
        self.logger.info("Pipeline cargado exitosamente desde: %s", self.model_artifact_path)

    def predict_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la inferencia diagnóstica individual en tiempo real para un paciente.

        Args:
            patient_dict: Diccionario con las mediciones clínicas del paciente.

        Returns:
            Diccionario estructurado con probabilidad, categoría de riesgo y factores de alerta.
        """
        if not self.is_trained:
            self.load_pipeline()

        # Construir DataFrame unitario
        df_single = pd.DataFrame([patient_dict])

        # Transformación
        X_trans = self.preprocessor.transform(df_single)

        # Inferencia
        prob = float(self.fitted_model.predict_proba(X_trans)[0, 1])
        prediction = int(prob >= 0.5)

        # Categorización clínica del riesgo
        if prob < 0.30:
            risk_category = "Bajo Riesgo"
            recommendation = "Manejo ambulatorio estándar y monitoreo rutinario."
            color = "#28a745"  # Verde
        elif prob < 0.60:
            risk_category = "Riesgo Moderado"
            recommendation = "Seguimiento médico estrecho y ajuste de farmacoterapia."
            color = "#ffc107"  # Amarillo
        else:
            risk_category = "Alto Riesgo Crítico"
            recommendation = "Evaluación prioritaria por cardiología / Posible hospitalización."
            color = "#dc3545"  # Rojo

        # Identificación de biomarcadores de alerta clínica
        alert_factors: List[str] = []
        if patient_dict.get("ejection_fraction", 50) < 30:
            alert_factors.append("Fracción de eyección severamente deprimida (<30%)")
        if patient_dict.get("serum_creatinine", 1.0) > 1.5:
            alert_factors.append("Creatinina sérica elevada (>1.5 mg/dL, indicativo de disfunción renal)")
        if patient_dict.get("serum_sodium", 137) < 135:
            alert_factors.append("Hiponatremia detectada (Sodio sérico <135 mEq/L)")
        if patient_dict.get("age", 50) >= 70:
            alert_factors.append("Edad avanzada del paciente (≥70 años)")

        return {
            "probability_mortality": prob,
            "risk_percentage": round(prob * 100, 2),
            "binary_prediction": prediction,
            "risk_category": risk_category,
            "clinical_recommendation": recommendation,
            "status_color": color,
            "alert_factors": alert_factors,
        }
