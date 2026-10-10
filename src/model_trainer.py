"""
Módulo de Entrenamiento, Validación Cruzada y Optimización de Modelos de IA.
Implementa clases POO para experimentación competitiva entre algoritmos
(Regresión Logística, Random Forest, Gradient Boosting) con validación
cruzada estratificada y selección basada en métricas clínicas.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.config import ModelExperimentConfig, setup_logger


class BaseModelTrainer(ABC):
    """
    Clase base abstracta que define la interfaz para el entrenamiento,
    validación y optimización de modelos predictivos.
    """

    def __init__(self, experiment_config: Optional[ModelExperimentConfig] = None) -> None:
        """
        Inicializa el entrenador base.

        Args:
            experiment_config: Configuración de particionado y evaluación.
        """
        self.config = experiment_config or ModelExperimentConfig()
        self.logger = setup_logger(self.__class__.__name__)
        self.best_estimator: Optional[BaseEstimator] = None
        self.best_estimator_name: str = ""
        self.benchmark_results: Dict[str, Dict[str, float]] = {}

    @abstractmethod
    def benchmark_models(
        self, X: np.ndarray, y: np.ndarray
    ) -> pd.DataFrame:
        """Compara múltiples algoritmos bajo validación cruzada estratificada."""
        pass

    @abstractmethod
    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, model_name: str
    ) -> BaseEstimator:
        """Ajusta hiperparámetros del modelo seleccionado mediante búsqueda en malla."""
        pass


class ClassificationModelTrainer(BaseModelTrainer):
    """
    Entrenador especializado en clasificación clínica para pronóstico de insuficiencia cardíaca.
    """

    def __init__(self, experiment_config: Optional[ModelExperimentConfig] = None) -> None:
        """Inicializa el entrenador con el catálogo de modelos candidatos."""
        super().__init__(experiment_config)
        self.candidate_models: Dict[str, BaseEstimator] = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
                class_weight="balanced",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                random_state=self.config.random_state,
                class_weight="balanced",
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                random_state=self.config.random_state,
            ),
        }

    def benchmark_models(
        self, X: np.ndarray, y: np.ndarray
    ) -> pd.DataFrame:
        """
        Ejecuta validación cruzada estratificada (k=5) sobre todos los modelos candidatos.
        Evalúa múltiples métricas: ROC-AUC, F1-Score, Recall (Sensibilidad) y Accuracy.

        Args:
            X: Matriz de características preprocesada.
            y: Vector objetivo binario.

        Returns:
            pd.DataFrame con el resumen comparativo de medias y desviaciones estándar.
        """
        self.logger.info(
            "Iniciando benchmarking con %d-Fold Stratified CV sobre %d candidatos...",
            self.config.n_splits_cv,
            len(self.candidate_models),
        )

        cv = StratifiedKFold(
            n_splits=self.config.n_splits_cv,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scoring = {
            "roc_auc": "roc_auc",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
            "accuracy": "accuracy",
        }

        results_list: List[Dict[str, Any]] = []

        for name, model in self.candidate_models.items():
            self.logger.info("Evaluando modelo candidato: %s", name)
            scores = cross_validate(
                model, X, y, cv=cv, scoring=scoring, return_train_score=False
            )

            metrics_summary = {
                "Model": name,
                "ROC_AUC_Mean": float(np.mean(scores["test_roc_auc"])),
                "ROC_AUC_Std": float(np.std(scores["test_roc_auc"])),
                "F1_Mean": float(np.mean(scores["test_f1"])),
                "F1_Std": float(np.std(scores["test_f1"])),
                "Recall_Mean": float(np.mean(scores["test_recall"])),
                "Recall_Std": float(np.std(scores["test_recall"])),
                "Accuracy_Mean": float(np.mean(scores["test_accuracy"])),
            }
            results_list.append(metrics_summary)
            self.benchmark_results[name] = metrics_summary

            self.logger.info(
                "[%s] ROC-AUC: %.4f (+/- %.4f) | Recall: %.4f | F1: %.4f",
                name,
                metrics_summary["ROC_AUC_Mean"],
                metrics_summary["ROC_AUC_Std"],
                metrics_summary["Recall_Mean"],
                metrics_summary["F1_Mean"],
            )

        benchmark_df = pd.DataFrame(results_list).sort_values(
            by="ROC_AUC_Mean", ascending=False
        )
        return benchmark_df

    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, model_name: str = "RandomForest"
    ) -> BaseEstimator:
        """
        Aplica GridSearchCV sobre el modelo seleccionado para maximizar ROC-AUC clínico.

        Args:
            X: Matriz de características preprocesada.
            y: Vector objetivo.
            model_name: Nombre del modelo a optimizar ('RandomForest', 'GradientBoosting' o 'LogisticRegression').

        Returns:
            Mejor estimador ajustado.
        """
        self.logger.info("Iniciando optimización de hiperparámetros para: %s", model_name)

        param_grids = {
            "RandomForest": {
                "n_estimators": [100, 150, 200],
                "max_depth": [4, 6, 8, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2, 4],
            },
            "GradientBoosting": {
                "n_estimators": [100, 150],
                "learning_rate": [0.03, 0.08, 0.15],
                "max_depth": [3, 4, 5],
                "subsample": [0.8, 1.0],
            },
            "LogisticRegression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        }

        if model_name not in param_grids:
            raise ValueError(f"Modelo '{model_name}' no soportado para optimización de hiperparámetros.")

        base_estimator = self.candidate_models[model_name]
        cv = StratifiedKFold(
            n_splits=self.config.n_splits_cv,
            shuffle=True,
            random_state=self.config.random_state,
        )

        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grids[model_name],
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            verbose=0,
        )

        grid_search.fit(X, y)
        self.best_estimator = grid_search.best_estimator_
        self.best_estimator_name = model_name

        self.logger.info(
            "Optimización finalizada para [%s]. Mejor ROC-AUC CV: %.4f | Mejores Hiperparámetros: %s",
            model_name,
            grid_search.best_score_,
            grid_search.best_params_,
        )

        return self.best_estimator
