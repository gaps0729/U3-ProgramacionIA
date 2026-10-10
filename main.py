"""
Punto de Entrada Principal (CLI) del Pipeline de Inteligencia Artificial.
Permite ejecutar el ciclo completo de ingesta, entrenamiento, evaluación y
persistencia de modelos de pronóstico cardiovascular.
"""

import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import setup_logger  # noqa: E402
from src.pipeline import PredictivePipeline  # noqa: E402


def main() -> None:
    """Función principal para orquestar la ejecución por consola."""
    logger = setup_logger("MainRunner")
    logger.info("Iniciando ejecución del Pipeline de Inteligencia Artificial (UDI)...")

    try:
        pipeline = PredictivePipeline()
        metrics = pipeline.train_and_evaluate()

        print("\n" + "=" * 60)
        print("RESULTADOS FINALES DE EVALUACIÓN (CONJUNTO DE PRUEBA)")
        print("=" * 60)
        print(f"Modelo Ganador:        {metrics['model_name']}")
        print(f"Accuracy Global:       {metrics['accuracy'] * 100:.2f}%")
        print(f"Sensibilidad (Recall): {metrics['sensitivity_recall'] * 100:.2f}%")
        print(f"Especificidad:         {metrics['specificity'] * 100:.2f}%")
        print(f"Precisión:             {metrics['precision'] * 100:.2f}%")
        print(f"F1-Score:              {metrics['f1_score']:.4f}")
        print(f"Área Bajo Curva ROC:   {metrics['roc_auc']:.4f}")
        print(f"Área Bajo Curva PR:    {metrics['pr_auc']:.4f}")
        print(f"Brier Score Loss:      {metrics['brier_score']:.4f}")
        print("=" * 60)
        print("Matriz de Confusión:")
        cm = metrics["confusion_matrix"]
        print(f"  Verdaderos Negativos (TN): {cm['true_negatives']}")
        print(f"  Falsos Positivos    (FP): {cm['false_positives']}")
        print(f"  Falsos Negativos    (FN): {cm['false_negatives']}")
        print(f"  Verdaderos Positivos(TP): {cm['true_positives']}")
        print("=" * 60)
        print("Pipeline finalizado exitosamente. Artefactos y gráficos guardados en models/ y docs/\n")

    except Exception as exc:
        logger.exception("Error durante la ejecución del pipeline: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
