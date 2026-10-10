"""
Módulo de Evaluación Analítica, Diagnóstico y Explicabilidad de Modelos.
Calcula métricas clínicas integrales (Recall, Precision, ROC-AUC, PR-AUC,
Brier Score), genera matrices de confusión y gráficos de interpretabilidad.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from src.config import METRICS_PATH, setup_logger

matplotlib.use("Agg")  # Backend no interactivo para entornos de servidor


class ModelEvaluator:
    """
    Evaluador clínico riguroso para modelos de pronóstico médico.
    """

    def __init__(self, metrics_output_path: Path = METRICS_PATH) -> None:
        """
        Inicializa el evaluador clínico.

        Args:
            metrics_output_path: Ruta para persistir el reporte analítico en formato JSON.
        """
        self.metrics_path = metrics_output_path
        self.logger = setup_logger(self.__class__.__name__)
        self.last_metrics: Dict[str, Any] = {}

    def evaluate_test_set(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "BestModel",
    ) -> Dict[str, Any]:
        """
        Calcula una batería completa de métricas estadísticas y clínicas.

        Args:
            y_true: Etiquetas reales de prueba.
            y_pred: Predicciones binarias (0 o 1).
            y_proba: Probabilidades continuas asignadas a la clase positiva.
            model_name: Nombre descriptivo del modelo evaluado.

        Returns:
            Diccionario estructurado con todas las métricas calculadas.
        """
        self.logger.info("Iniciando evaluación clínica integral sobre el conjunto de prueba para [%s]...", model_name)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        brier = brier_score_loss(y_true, y_proba)
        accuracy = accuracy_score(y_true, y_pred)

        metrics: Dict[str, Any] = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "sensitivity_recall": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

        self.last_metrics = metrics
        self.logger.info(
            "[%s] Desempeño: Sensibilidad (Recall)=%.4f, Especificidad=%.4f, ROC-AUC=%.4f, F1=%.4f, Brier=%.4f",
            model_name,
            sensitivity,
            specificity,
            roc_auc,
            f1,
            brier,
        )

        # Persistir a disco en formato JSON
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info("Métricas guardadas satisfactoriamente en: %s", self.metrics_path)

        return metrics

    def generate_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        feature_names: List[str],
        feature_importances: Optional[np.ndarray],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Genera y guarda en disco los gráficos analíticos requeridos:
        1. Curva ROC con valor de AUC.
        2. Matriz de confusión visualizada con mapa de calor (heatmap).
        3. Gráfico de barras de importancia de características biomédicas.

        Args:
            y_true: Vector real.
            y_pred: Predicciones de clase.
            y_proba: Probabilidades predichas.
            feature_names: Lista ordenada de nombres de predictores.
            feature_importances: Vector numérico de coeficientes o importancias de árbol.
            output_dir: Carpeta de destino para las imágenes.

        Returns:
            Diccionario que mapea el nombre del gráfico a su ruta física generada.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        # 1. Curva ROC
        roc_path = output_dir / "roc_curve.png"
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Modelo Clínico (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Clasificador Aleatorio")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=11)
        plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)", fontsize=11)
        plt.title("Curva ROC - Predicción de Mortalidad Cardíaca", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()
        paths["roc_curve"] = roc_path

        # 2. Matriz de Confusión
        cm_path = output_dir / "confusion_matrix.png"
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Sobrevivió (0)", "Falleció (1)"],
            yticklabels=["Sobrevivió (0)", "Falleció (1)"],
            cbar=False,
        )
        plt.xlabel("Diagnóstico Predicho por el Modelo", fontsize=11)
        plt.ylabel("Evolución Real del Paciente", fontsize=11)
        plt.title("Matriz de Confusión Clínica", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        paths["confusion_matrix"] = cm_path

        # 3. Importancia de Características (Feature Importance)
        if feature_importances is not None and len(feature_importances) == len(feature_names):
            fi_path = output_dir / "feature_importance.png"
            indices = np.argsort(feature_importances)

            plt.figure(figsize=(8, 6))
            plt.barh(
                range(len(indices)),
                feature_importances[indices],
                color="#2ca02c",
                align="center",
            )
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
            plt.xlabel("Importancia Relativa del Biomarcador", fontsize=11)
            plt.title("Jerarquía de Biomarcadores en el Pronóstico Clínico", fontsize=13, fontweight="bold")
            plt.grid(True, axis="x", linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(fi_path, dpi=300)
            plt.close()
            paths["feature_importance"] = fi_path

        self.logger.info("Gráficos diagnósticos generados exitosamente en: %s", output_dir)
        return paths
