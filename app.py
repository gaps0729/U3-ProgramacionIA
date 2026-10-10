"""
Dashboard Interactivo de Soporte a la Decisión Clínica con Gradio.
Permite inferencia en tiempo real, simulación de biomarcadores,
visualización de métricas analíticas y explicabilidad diagnóstica.
Maestría en Ciencia de Datos e Inteligencia Artificial - UDI.
"""

import json
from typing import Tuple
import gradio as gr

from src.config import BASE_DIR, METRICS_PATH, setup_logger
from src.pipeline import PredictivePipeline

logger = setup_logger("GradioDashboard")

# Inicializar y cargar el pipeline serializado
pipeline = PredictivePipeline()
try:
    pipeline.load_pipeline()
    logger.info("Pipeline cargado exitosamente en Gradio.")
except Exception as err:
    logger.warning("Pipeline aún no entrenado o archivo no encontrado: %s", err)


def predict_clinical_risk(
    age: float,
    anaemia: str,
    creatinine_phosphokinase: float,
    diabetes: str,
    ejection_fraction: float,
    high_blood_pressure: str,
    platelets: float,
    serum_creatinine: float,
    serum_sodium: float,
    sex: str,
    smoking: str,
    time: float,
) -> Tuple[str, str, str, str]:
    """
    Función de inferencia invocada por la interfaz interactiva de Gradio.

    Returns:
        Tupla con (HTML de Riesgo, Porcentaje, Recomendación Clínica, Alertas de Biomarcadores).
    """
    patient_payload = {
        "age": float(age),
        "anaemia": 1 if "Sí" in anaemia else 0,
        "creatinine_phosphokinase": float(creatinine_phosphokinase),
        "diabetes": 1 if "Sí" in diabetes else 0,
        "ejection_fraction": float(ejection_fraction),
        "high_blood_pressure": 1 if "Sí" in high_blood_pressure else 0,
        "platelets": float(platelets),
        "serum_creatinine": float(serum_creatinine),
        "serum_sodium": float(serum_sodium),
        "sex": 1 if "Masculino" in sex else 0,
        "smoking": 1 if "Sí" in smoking else 0,
        "time": float(time),
    }

    result = pipeline.predict_patient(patient_payload)

    color = result["status_color"]
    category = result["risk_category"]
    prob_pct = result["risk_percentage"]
    recommendation = result["clinical_recommendation"]
    alerts = result["alert_factors"]

    # Generación de tarjetas HTML estilizadas
    risk_html = f"""
    <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 18px; border-radius: 8px;">
        <h2 style="color: {color}; margin: 0 0 8px 0; font-family: sans-serif;">
            Diagnóstico: {category}
        </h2>
        <p style="font-size: 16px; margin: 0; color: #333;">
            Probabilidad estimada de mortalidad a seguimiento: <strong>{prob_pct}%</strong>
        </p>
    </div>
    """

    alert_items = "".join([f"<li>⚠️ <strong>{item}</strong></li>" for item in alerts])
    if not alert_items:
        alert_items = "<li>✅ Parámetros críticos dentro de umbrales controlados.</li>"

    alerts_html = f"""
    <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 14px; border-radius: 8px;">
        <h4 style="margin: 0 0 10px 0; color: #495057;">Biomarcadores Críticos y Factores de Riesgo:</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #212529;">
            {alert_items}
        </ul>
    </div>
    """

    return risk_html, f"{prob_pct}%", recommendation, alerts_html


def load_metrics_summary() -> str:
    """Carga y formatea las métricas del modelo en una tabla Markdown."""
    if not METRICS_PATH.exists():
        return "⚠️ Métricas no disponibles. Ejecute `python main.py` para entrenar el modelo."

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = data.get("confusion_matrix", {})
    roc_val = data.get("roc_auc", 0)
    rec_val = data.get("sensitivity_recall", 0) * 100
    spec_val = data.get("specificity", 0) * 100
    prec_val = data.get("precision", 0) * 100
    f1_val = data.get("f1_score", 0)
    pr_val = data.get("pr_auc", 0)
    brier_val = data.get("brier_score", 0)

    markdown_table = (
        "### 📊 Desempeño Cuantitativo en el Conjunto de Prueba (Holdout Test Set)\n\n"
        "| Métrica Diagnóstica | Valor Obtenido | Relevancia Clínica / Interpretación |\n"
        "| :--- | :--- | :--- |\n"
        f"| **Modelo Seleccionado** | `{data.get('model_name')}` | Optimizado mediante Validación Cruzada |\n"
        f"| **Área Bajo Curva ROC (AUC)** | **{roc_val:.4f}** | Capacidad discriminatoria sobresaliente |\n"
        f"| **Sensibilidad / Recall** | **{rec_val:.2f}%** | Proporción de eventos críticos detectados |\n"
        f"| **Especificidad** | **{spec_val:.2f}%** | Capacidad de identificar pacientes estables |\n"
        f"| **Precisión** | **{prec_val:.2f}%** | Certeza ante una alerta emitida por el modelo |\n"
        f"| **F1-Score** | **{f1_val:.4f}** | Media armónica entre Sensibilidad y Precisión |\n"
        f"| **Área Bajo Curva PR (PR-AUC)** | **{pr_val:.4f}** | Precisión en desbalance de clases |\n"
        f"| **Brier Score (Calibración)** | **{brier_val:.4f}** | Calidad probabilística (óptimo cercano a 0) |\n\n"
        "#### Matriz de Confusión Clínica:\n"
        f"- **Verdaderos Negativos (TN):** {cm.get('true_negatives')} pacientes\n"
        f"- **Falsos Positivos (FP):** {cm.get('false_positives')} pacientes\n"
        f"- **Falsos Negativos (FN):** {cm.get('false_negatives')} pacientes *(Riesgo crítico mitigado)*\n"
        f"- **Verdaderos Positivos (TP):** {cm.get('true_positives')} pacientes\n"
    )
    return markdown_table


# ==============================================================================
# Construcción de la Interfaz Gradio
# ==============================================================================
docs_dir = BASE_DIR / "docs"
roc_img_path = docs_dir / "roc_curve.png"
cm_img_path = docs_dir / "confusion_matrix.png"
fi_img_path = docs_dir / "feature_importance.png"

with gr.Blocks(title="CDSS - Pronóstico de Insuficiencia Cardíaca (UDI)") as demo:
    gr.Markdown(
        """
        # 🩺 Sistema de Soporte a la Decisión Clínica (CDSS)
        ### **Predicción de Riesgo y Mortalidad en Insuficiencia Cardíaca con IA Explicable**
        *Maestría en Ciencia de Datos e Inteligencia Artificial — Universidad de Investigación y Desarrollo (UDI)*
        """
    )

    with gr.Tabs():
        # TAB 1: Simulador Clínico
        with gr.TabItem("🏥 Simulador Clínico Individual"):
            gr.Markdown(
                "Ajuste los biomarcadores clínicos y variables basales del paciente para calcular en tiempo real "
                "el riesgo de eventos adversos y recibir recomendaciones de manejo asistencial."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🧬 Variables Fisiológicas Principales")
                    age_input = gr.Slider(minimum=20, maximum=95, value=65, step=1, label="Edad (Años)")
                    ef_input = gr.Slider(
                        minimum=10, maximum=80, value=35, step=1,
                        label="Fracción de Eyección (%) [Normal: >50%]"
                    )
                    sc_input = gr.Slider(
                        minimum=0.4, maximum=9.5, value=1.4, step=0.1,
                        label="Creatinina Sérica (mg/dL) [Normal: 0.7 - 1.3]"
                    )
                    sod_input = gr.Slider(
                        minimum=110, maximum=150, value=136, step=1,
                        label="Sodio Sérico (mEq/L) [Normal: 135 - 145]"
                    )
                    cpk_input = gr.Slider(
                        minimum=20, maximum=5000, value=250, step=10,
                        label="Creatina Fosfocinasa (CPK mcg/L)"
                    )
                    plat_input = gr.Slider(
                        minimum=30000, maximum=700000, value=250000, step=5000,
                        label="Plaquetas (kiloplaquetas/mL)"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### 📋 Comorbilidades e Historial")
                    hbp_input = gr.Dropdown(
                        choices=["No (0)", "Sí (1)"], value="Sí (1)", label="Hipertensión Arterial"
                    )
                    diab_input = gr.Dropdown(
                        choices=["No (0)", "Sí (1)"], value="No (0)", label="Diabetes Mellitus"
                    )
                    anae_input = gr.Dropdown(
                        choices=["No (0)", "Sí (1)"], value="No (0)", label="Anemia"
                    )
                    smok_input = gr.Dropdown(
                        choices=["No (0)", "Sí (1)"], value="No (0)", label="Tabaquismo"
                    )
                    sex_input = gr.Dropdown(
                        choices=["Femenino (0)", "Masculino (1)"], value="Masculino (1)", label="Sexo Biológico"
                    )
                    time_input = gr.Slider(
                        minimum=4, maximum=300, value=90, step=1,
                        label="Tiempo de Seguimiento (Días)"
                    )

            predict_btn = gr.Button("🔍 Calcular Riesgo Clínico y Evaluar Paciente", variant="primary", size="lg")

            gr.Markdown("### 📋 Resultados del Diagnóstico Predictivo")
            with gr.Row():
                with gr.Column(scale=1):
                    output_html = gr.HTML(label="Clasificación de Riesgo")
                    output_prob = gr.Textbox(label="Probabilidad de Riesgo", interactive=False)
                with gr.Column(scale=1):
                    output_recommendation = gr.Textbox(
                        label="Recomendación Terapéutica / Protocolo Clínico", interactive=False, lines=2
                    )
                    output_alerts = gr.HTML(label="Factores de Alerta")

            # Casos de ejemplo preestablecidos
            ex1 = [
                45.0, "No (0)", 150.0, "No (0)", 55.0, "No (0)", 280000.0, 0.8, 140.0, "Masculino (1)", "No (0)", 220.0
            ]
            ex2 = [
                75.0, "Sí (1)", 582.0, "Sí (1)", 20.0, "Sí (1)", 150000.0, 3.2, 131.0, "Femenino (0)", "Sí (1)", 15.0
            ]
            ex3 = [
                60.0, "No (0)", 250.0, "Sí (1)", 38.0, "Sí (1)", 240000.0, 1.3, 137.0, "Masculino (1)", "No (0)", 110.0
            ]

            gr.Examples(
                examples=[ex1, ex2, ex3],
                inputs=[
                    age_input, anae_input, cpk_input, diab_input, ef_input, hbp_input,
                    plat_input, sc_input, sod_input, sex_input, smok_input, time_input,
                ],
                label="💡 Casos Clínicos de Demostración (Haga clic para autocompletar)",
            )

            predict_btn.click(
                fn=predict_clinical_risk,
                inputs=[
                    age_input, anae_input, cpk_input, diab_input, ef_input, hbp_input,
                    plat_input, sc_input, sod_input, sex_input, smok_input, time_input,
                ],
                outputs=[output_html, output_prob, output_recommendation, output_alerts],
            )

        # TAB 2: Analítica y Rendimiento del Modelo
        with gr.TabItem("📊 Analítica y Diagnóstico del Modelo"):
            gr.Markdown(load_metrics_summary)

            with gr.Row():
                if roc_img_path.exists():
                    gr.Image(value=str(roc_img_path), label="Curva ROC - Capacidad de Discriminación")
                if cm_img_path.exists():
                    gr.Image(value=str(cm_img_path), label="Matriz de Confusión Clínica")

            with gr.Row():
                if fi_img_path.exists():
                    gr.Image(value=str(fi_img_path), label="Jerarquía e Importancia de Biomarcadores")

        # TAB 3: Metodología y Proyecto Académico
        with gr.TabItem("📖 Ficha Técnica y Metodología"):
            gr.Markdown(
                """
                ### 🎓 Información Institucional
                - **Institución:** Universidad de Investigación y Desarrollo (UDI)
                - **Programa:** Maestría en Ciencia de Datos e Inteligencia Artificial
                - **Curso:** Programación con IA — Actividad Final (25%)
                - **Eje Temático:** Salud y Biología (Predicción de Insuficiencia Cardíaca)

                ### 🛠️ Arquitectura de Software y Buenas Prácticas
                - **Programación Orientada a Objetos (POO):** Clases abstractas (`BaseDataPipeline`,
                  `BaseFeatureEngineer`), encapsulamiento, polimorfismo y principios SOLID.
                - **Validación Sintáctica con Regex:** Sanitización obligatoria de cabeceras y tipos de datos
                  mediante el módulo estándar `re`.
                - **Observabilidad:** Sistema de logging estructurado rotativo con el módulo estándar `logging`.
                - **Prevención de Data Leakage:** Pipeline desacoplado donde la estandarización y estadísticas
                  solo se calculan sobre el conjunto de entrenamiento.
                - **Pruebas Automatizadas:** Suite de tests unitarios exhaustiva con `pytest`.

                ### 📚 Referencias Bibliográficas (Normas APA 7ma Edición)
                - Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart
                  failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision
                  Making*, 20(1), 16. https://doi.org/10.1186/s12911-020-1023-5
                - Gradio Team. (2024). *Gradio: Build & share delightful machine learning apps*. https://gradio.app/
                - Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
                  *Journal of Machine Learning Research*, 12, 2825–2830.
                """
            )

if __name__ == "__main__":
    logger.info("Iniciando servidor de Gradio...")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(), share=False)
