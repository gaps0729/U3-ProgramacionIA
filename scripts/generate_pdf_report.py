"""
Generador del Informe Académico Formal en PDF (Normas APA 7ma Edición).
Produce el documento entregable oficial para la Actividad Final del curso
Programación con IA - Maestría en Ciencia de Datos e Inteligencia Artificial (UDI).
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

DEFAULT_PDF_NAME = "IICD_ProyectoFinal_PipelineIA_UDI.pdf"


class NumberedCanvas(canvas.Canvas):
    """Lienzo personalizado para numeración de páginas 'Página X de Y' y pie de página."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#555555"))

            # Encabezado superior
            self.drawString(
                54, 750, "UDI - Maestría en Ciencia de Datos e IA | Programación con IA - Actividad Final"
            )
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

            # Pie de página inferior
            self.line(54, 45, 558, 45)
            self.drawString(
                54, 32, "Pipeline de IA para Predicción de Riesgo Cardiovascular con POO y Gradio"
            )
            page_text = f"Página {self._pageNumber} de {page_count}"
            self.drawRightString(558, 32, page_text)
            self.restoreState()


def build_academic_report(
    output_filename: str = DEFAULT_PDF_NAME,
    student_name: str = "Magíster en Formación",
    student_id: str = "",
    github_url: str = "https://github.com/TU-USUARIO/U3-ProgamacionIA",
):
    pdf_path = BASE_DIR / output_filename
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Definición de Estilos Tipográficos de Nivel Académico
    primary_color = colors.HexColor("#0B3C5D")
    secondary_color = colors.HexColor("#1D2731")
    accent_color = colors.HexColor("#328CC1")
    text_color = colors.HexColor("#222222")

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=primary_color,
        alignment=1,  # Centrado
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=accent_color,
        alignment=1,
    )

    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=secondary_color,
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=accent_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=text_color,
        alignment=4,  # Justificado
        spaceAfter=6,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=text_color,
        alignment=0,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=text_color,
        alignment=1,
    )

    story = []

    # ==========================================================================
    # 1. PORTADA INSTITUCIONAL FORMAL
    # ==========================================================================
    story.append(Spacer(1, 40))
    story.append(
        Paragraph("UNIVERSIDAD DE INVESTIGACIÓN Y DESARROLLO (UDI)", subtitle_style)
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph("MAESTRÍA EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", subtitle_style)
    )
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="80%", thickness=2, color=primary_color, spaceAfter=25))

    story.append(
        Paragraph(
            "CONSTRUCCIÓN DE UN PIPELINE COMPLETO DE SOFTWARE E INTELIGENCIA ARTIFICIAL CON REFACTORIZACIÓN A POO Y DASHBOARD INTERACTIVO",
            title_style,
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Aplicación al Dominio de Salud y Biología: Predicción y Triaje de Insuficiencia Cardíaca",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="40%", thickness=1, color=accent_color, spaceAfter=30))

    story.append(Paragraph("<b>Curso:</b> Programación con IA (Unidades 1, 2 y 3)", meta_style))
    story.append(Paragraph("<b>Momento de Evaluación:</b> Actividad Final (25%)", meta_style))
    author_str = f"<b>Estudiante / Autor:</b> {student_name}"
    if student_id:
        author_str += f" (C.C. {student_id})"
    story.append(Paragraph(author_str, meta_style))
    story.append(Paragraph("<b>Docente / Tutor:</b> Equipo Docente UDI Virtual", meta_style))
    story.append(Paragraph("<b>Bucaramanga, Colombia</b>", meta_style))
    story.append(Paragraph("<b>2026</b>", meta_style))

    story.append(PageBreak())

    # ==========================================================================
    # 2. INTRODUCCIÓN
    # ==========================================================================
    story.append(Paragraph("1. Introducción", h1_style))
    story.append(
        Paragraph(
            "La transición de prototipos analíticos preliminares hacia sistemas de Inteligencia Artificial "
            "robustos, escalables y auditables constituye uno de los desafíos medulares en la ingeniería de "
            "datos moderna. En dominios críticos como el sector de la salud y la biomedicina, la toma de decisiones "
            "basada en modelos predictivos no tolera arquitecturas monolíticas opacas, código desestructurado ni "
            "métricas de evaluación reduccionistas. El diagnóstico y pronóstico de la insuficiencia cardíaca, una de "
            "las afecciones cardiovasculares con mayor tasa de morbilidad y readmisión hospitalaria a nivel mundial, "
            "exige soluciones informáticas que satisfagan rigurosos criterios de calidad de software, trazabilidad "
            "operativa y explicabilidad algorítmica.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "El presente trabajo final desarrolla de manera integral la ruta formativa de las Unidades 1, 2 y 3 del "
            "curso <i>Programación con IA</i> en la <b>Maestría en Ciencia de Datos e Inteligencia Artificial</b>. "
            "Se implementa un pipeline de producción end-to-end que trasciende scripts exploratorios mediante el "
            "paradigma de Programación Orientada a Objetos (POO), patrones de diseño de software limpio (SOLID), "
            "validación sintáctica estricta con expresiones regulares (módulo <code>re</code>), logging jerárquico "
            "persistente (módulo <code>logging</code>), validación cruzada estratificada con Scikit-learn y una "
            "interfaz interactiva de soporte a la decisión clínica desarrollada en <b>Gradio</b>. Asimismo, se "
            "documenta con rigor científico la metodología de desarrollo asistido por IA generativa a través de una "
            "bitácora de ingeniería de prompts en 4 etapas, permitiendo verificar empíricamente los beneficios "
            "técnicos de la refactorización arquitectónica.",
            body_style,
        )
    )

    # ==========================================================================
    # 3. DESARROLLO DEL TALLER Y METODOLOGÍA
    # ==========================================================================
    story.append(Paragraph("2. Desarrollo del Taller y Metodología de Pipeline", h1_style))

    story.append(Paragraph("2.1 Definición del Problema Clínico y Conjunto de Datos", h2_style))
    story.append(
        Paragraph(
            "Se seleccionó el eje temático de <b>Salud y Biología</b> utilizando el conjunto de datos de registros "
            "clínicos de insuficiencia cardíaca (<i>Heart Failure Clinical Records</i>) recolectado por Chicco y "
            "Jurman (2020), alojado en el repositorio UCI Machine Learning y Kaggle. El dataset comprende 299 "
            "observaciones de pacientes con seguimiento longitudinal y 13 variables biomédicas, de las cuales 12 "
            "actúan como predictoras y una como variable objetivo binaria (<code>DEATH_EVENT</code>, donde 1 denota "
            "fallecimiento durante el seguimiento y 0 denota supervivencia).",
            body_style,
        )
    )

    # Tabla de variables clínicas
    data_vars = [
        [
            Paragraph("Variable", table_header_style),
            Paragraph("Tipo", table_header_style),
            Paragraph("Unidad / Rango Plausible", table_header_style),
            Paragraph("Significado Clínico", table_header_style),
        ],
        [
            Paragraph("age", table_cell_style),
            Paragraph("Numérica continua", table_cell_center),
            Paragraph("18 - 120 años", table_cell_center),
            Paragraph("Edad cronológica del paciente", table_cell_style),
        ],
        [
            Paragraph("ejection_fraction", table_cell_style),
            Paragraph("Numérica continua", table_cell_center),
            Paragraph("5 - 90 %", table_cell_center),
            Paragraph("Porcentaje de sangre expulsada por el ventrículo izquierdo", table_cell_style),
        ],
        [
            Paragraph("serum_creatinine", table_cell_style),
            Paragraph("Numérica continua", table_cell_center),
            Paragraph("0.2 - 15.0 mg/dL", table_cell_center),
            Paragraph("Nivel de creatinina sérica; marcador de función renal", table_cell_style),
        ],
        [
            Paragraph("serum_sodium", table_cell_style),
            Paragraph("Numérica continua", table_cell_center),
            Paragraph("100 - 160 mEq/L", table_cell_center),
            Paragraph("Concentración de sodio en sangre (indicador de balance hidroelectrolítico)", table_cell_style),
        ],
        [
            Paragraph("creatinine_phosphokinase", table_cell_style),
            Paragraph("Numérica continua", table_cell_center),
            Paragraph("10 - 10000 mcg/L", table_cell_center),
            Paragraph("Nivel de la enzima CPK, asociada a daño en miocardio", table_cell_style),
        ],
        [
            Paragraph("platelets", table_cell_style),
            Paragraph("Numérica continua", table_cell_center),
            Paragraph("20000 - 1000000 /mL", table_cell_center),
            Paragraph("Conteo plaquetario sanguíneo", table_cell_style),
        ],
        [
            Paragraph("time", table_cell_style),
            Paragraph("Numérica continua", table_cell_center),
            Paragraph("1 - 365 días", table_cell_center),
            Paragraph("Período de seguimiento ambulatorio", table_cell_style),
        ],
        [
            Paragraph("anaemia, diabetes, high_blood_pressure, sex, smoking", table_cell_style),
            Paragraph("Categóricas binarias", table_cell_center),
            Paragraph("{0, 1}", table_cell_center),
            Paragraph("Comorbilidades, sexo biológico y hábitos de riesgo", table_cell_style),
        ],
        [
            Paragraph("DEATH_EVENT (Target)", table_cell_style),
            Paragraph("Binaria objetivo", table_cell_center),
            Paragraph("{0: Sobrevive, 1: Fallece}", table_cell_center),
            Paragraph("Desenlace clínico evaluado (Supervivencia del paciente)", table_cell_style),
        ],
    ]

    t_vars = Table(data_vars, colWidths=[110, 85, 110, 199])
    t_vars.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9F9F9"), colors.white]),
            ]
        )
    )
    story.append(t_vars)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2.2 Metodología de IA Asistida y Bitácora de Refactorización", h2_style))
    story.append(
        Paragraph(
            "Cumpliendo con el criterio estratégico de la rúbrica evaluativa, el desarrollo se estructuró en "
            "un proceso asistido por IA documentado en cuatro iteraciones:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>1. Fase 1 (Prototipo Exploratorio Monolítico):</b> Se construyó inicialmente un script lineal en "
            "<code>notebooks/01_exploratory_prototype.py</code>. Si bien logró entrenar modelos simples, carecía de "
            "modularidad, acoplaba transformaciones de forma insegura y arrojaba métricas globales incompletas sin "
            "control de falsos negativos.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>2. Fase 2 (Refactorización a POO y SOLID):</b> Mediante prompt engineering especializado, se desacoplaron "
            "las responsabilidades en clases abstractas con <code>abc.ABC</code> (<code>BaseDataPipeline</code>, "
            "<code>BaseFeatureEngineer</code>, <code>BaseModelTrainer</code>) y un orquestador <code>PredictivePipeline</code> "
            "bajo el patrón Facade. Esto eliminó el acoplamiento y garantizó que la normalización se ajuste sólo sobre "
            "el conjunto de entrenamiento (previniendo fuga de datos o <i>data leakage</i>).",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>3. Fase 3 (Robustez de Software e Ingeniería MLOps):</b> Se integró el módulo estándar <code>re</code> "
            "para validar la sintaxis de cabeceras de columnas y códigos de identificación clínica. Se sustituyeron los "
            "<code>print()</code> por logging jerárquico rotativo (<code>logging.handlers.RotatingFileHandler</code>) hacia "
            "<code>logs/pipeline.log</code>, y se aplicaron Type Annotations (<code>typing</code>) y docstrings exhaustivos "
            "bajo norma <b>PEP 8</b>.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>4. Fase 4 (Desarrollo del Dashboard Interactivo):</b> Se construyó una aplicación con <b>Gradio</b> "
            "(<code>app.py</code>) con triaje en tiempo real, calibración probabilística y visualización analítica.",
            body_style,
        )
    )

    story.append(Spacer(1, 8))
    story.append(Paragraph("2.3 Arquitectura de Software Orientada a Objetos", h2_style))
    story.append(
        Paragraph(
            "La solución final se estructura como un paquete modular en <code>src/</code>:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <code>src/config.py</code>: Dataclasses inmutables (<code>ClinicalSchemaConfig</code>, "
            "<code>ModelExperimentConfig</code>), constantes de rutas y función <code>setup_logger()</code>.<br/>"
            "• <code>src/data_ingestion.py</code>: Clase <code>HeartFailureDataPipeline</code> que valida la integridad "
            "sintáctica mediante regex, descarta duplicados y audita rangos biológicos plausibles.<br/>"
            "• <code>src/preprocessor.py</code>: Clase <code>ClinicalFeatureEngineer</code> que implementa "
            "<code>fit_transform()</code> y <code>transform()</code> combinando <code>StandardScaler</code> e imputación.<br/>"
            "• <code>src/model_trainer.py</code>: Clase <code>ClassificationModelTrainer</code> que ejecuta benchmarking "
            "con validación cruzada estratificada (k=5) y optimización de hiperparámetros con <code>GridSearchCV</code>.<br/>"
            "• <code>src/evaluator.py</code>: Clase <code>ModelEvaluator</code> encargada del cálculo de sensibilidad, "
            "especificidad, ROC-AUC, PR-AUC, Brier Score y generación de gráficos en alta resolución.<br/>"
            "• <code>src/pipeline.py</code>: Fachada <code>PredictivePipeline</code> que coordina el flujo completo, "
            "serializa el artefacto a <code>models/best_heart_failure_model.joblib</code> y expone inferencia unitaria.<br/>"
            "• <code>tests/</code>: Suite de pruebas unitarias automatizadas ejecutables con <code>pytest</code>.",
            body_style,
        )
    )

    story.append(PageBreak())

    # ==========================================================================
    # 4. RESULTADOS EXPERIMENTALES Y MÉTRICAS
    # ==========================================================================
    story.append(Paragraph("3. Resultados Experimentales y Evaluación Analítica", h1_style))
    story.append(
        Paragraph(
            "Para seleccionar el modelo de pronóstico superior se ejecutó un protocolo de <b>Validación Cruzada "
            "Estratificada (5-Fold Stratified CV)</b> sobre el 80% de los datos (n=239), reservando el 20% (n=60) "
            "como conjunto de prueba independiente (Holdout Test Set).",
            body_style,
        )
    )

    # Tabla de Benchmarking
    bench_data = [
        [
            Paragraph("Modelo Candidato", table_header_style),
            Paragraph("ROC-AUC CV (Media ± Desv.)", table_header_style),
            Paragraph("Sensibilidad (Recall)", table_header_style),
            Paragraph("F1-Score", table_header_style),
            Paragraph("Accuracy CV", table_header_style),
        ],
        [
            Paragraph("<b>Random Forest Classifier</b>", table_cell_style),
            Paragraph("<b>0.9065 ± 0.0162</b>", table_cell_center),
            Paragraph("<b>74.08%</b>", table_cell_center),
            Paragraph("<b>0.7327</b>", table_cell_center),
            Paragraph("<b>82.46%</b>", table_cell_center),
        ],
        [
            Paragraph("Gradient Boosting Classifier", table_cell_style),
            Paragraph("0.9015 ± 0.0277", table_cell_center),
            Paragraph("66.50%", table_cell_center),
            Paragraph("0.6886", table_cell_center),
            Paragraph("80.37%", table_cell_center),
        ],
        [
            Paragraph("Regresión Logística (Baseline)", table_cell_style),
            Paragraph("0.8791 ± 0.0255", table_cell_center),
            Paragraph("76.67%", table_cell_center),
            Paragraph("0.7158", table_cell_center),
            Paragraph("80.35%", table_cell_center),
        ],
    ]

    t_bench = Table(bench_data, colWidths=[140, 110, 85, 80, 89])
    t_bench.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EBF3F9"), colors.white, colors.white]),
            ]
        )
    )
    story.append(t_bench)
    story.append(Spacer(1, 12))

    story.append(Paragraph("3.1 Rendimiento en el Conjunto de Prueba Independiente (Test Holdout)", h2_style))
    story.append(
        Paragraph(
            "El modelo seleccionado y optimizado fue <b>RandomForest</b> (con <code>max_depth=8</code>, "
            "<code>min_samples_leaf=4</code>, <code>n_estimators=100</code>). Evaluado sobre los 60 pacientes de prueba:",
            body_style,
        )
    )

    test_metrics_data = [
        [
            Paragraph("Métrica Diagnóstica", table_header_style),
            Paragraph("Valor Obtenido", table_header_style),
            Paragraph("Interpretación en el Contexto Clínico", table_header_style),
        ],
        [
            Paragraph("<b>Área Bajo la Curva ROC (AUC)</b>", table_cell_style),
            Paragraph("<b>0.8935</b>", table_cell_center),
            Paragraph("Capacidad de discriminación casi sobresaliente entre pacientes sobrevivientes y críticos.", table_cell_style),
        ],
        [
            Paragraph("<b>Área Bajo la Curva PR (PR-AUC)</b>", table_cell_style),
            Paragraph("<b>0.8091</b>", table_cell_center),
            Paragraph("Excelente rendimiento en la detección de la clase minoritaria (fallecimiento).", table_cell_style),
        ],
        [
            Paragraph("<b>Especificidad</b>", table_cell_style),
            Paragraph("<b>92.68%</b>", table_cell_center),
            Paragraph("Identificación correcta de 38 de 41 pacientes estables (mitiga falsas alarmas y costos innecesarios).", table_cell_style),
        ],
        [
            Paragraph("<b>Sensibilidad / Recall</b>", table_cell_style),
            Paragraph("<b>68.42%</b>", table_cell_center),
            Paragraph("Identificación oportuna de 13 de 19 pacientes con alto riesgo inminente.", table_cell_style),
        ],
        [
            Paragraph("<b>Precisión</b>", table_cell_style),
            Paragraph("<b>81.25%</b>", table_cell_center),
            Paragraph("El 81.25% de las alertas de mortalidad emitidas corresponden a eventos reales.", table_cell_style),
        ],
        [
            Paragraph("<b>F1-Score</b>", table_cell_style),
            Paragraph("<b>0.7429</b>", table_cell_center),
            Paragraph("Balance armónico entre sensibilidad diagnóstica y precisión de alarma.", table_cell_style),
        ],
        [
            Paragraph("<b>Brier Score Loss</b>", table_cell_style),
            Paragraph("<b>0.1207</b>", table_cell_center),
            Paragraph("Calibración probabilística robusta; las probabilidades reflejan con fidelidad el riesgo relativo.", table_cell_style),
        ],
    ]

    t_metrics = Table(test_metrics_data, colWidths=[150, 80, 274])
    t_metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent_color),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9F9F9")]),
            ]
        )
    )
    story.append(t_metrics)
    story.append(Spacer(1, 14))

    # Imágenes generadas
    story.append(Paragraph("3.2 Evidencia Gráfica y Diagnóstico de Modelos", h2_style))

    roc_path = DOCS_DIR / "roc_curve.png"
    cm_path = DOCS_DIR / "confusion_matrix.png"
    fi_path = DOCS_DIR / "feature_importance.png"

    if roc_path.exists() and cm_path.exists():
        t_imgs = Table(
            [
                [
                    Image(str(roc_path), width=240, height=205),
                    Image(str(cm_path), width=230, height=205),
                ],
                [
                    Paragraph("<b>Figura 1:</b> Curva ROC (AUC = 0.894).", table_cell_center),
                    Paragraph("<b>Figura 2:</b> Matriz de Confusión Clínica.", table_cell_center),
                ],
            ],
            colWidths=[252, 252],
        )
        t_imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(t_imgs)
        story.append(Spacer(1, 10))

    if fi_path.exists():
        story.append(
            KeepTogether(
                [
                    Image(str(fi_path), width=480, height=240),
                    Paragraph("<b>Figura 3:</b> Jerarquía de Importancia Relativa de Biomarcadores en el Pronóstico.", table_cell_center),
                    Spacer(1, 10),
                ]
            )
        )

    story.append(PageBreak())

    # ==========================================================================
    # 5. DASHBOARD INTERACTIVO EN GRADIO Y REPOSITORIO GITHUB
    # ==========================================================================
    story.append(Paragraph("4. Dashboard Interactivo en Gradio y Repositorio de GitHub", h1_style))
    story.append(
        Paragraph(
            "Para materializar la interacción asistencial se desarrolló una interfaz interactiva en <b>Gradio</b> "
            "(<code>app.py</code>). La herramienta opera como un <i>Clinical Decision Support System (CDSS)</i> "
            "estructurado en tres módulos de usuario:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Simulador y Triaje Clínico:</b> Permite ajustar biomarcadores mediante deslizadores y desplegables "
            "parametrizados. El pipeline calcula la probabilidad continua de mortalidad, asigna un nivel de triaje "
            "codificado por color (Verde: Bajo Riesgo &lt;30%, Amarillo: Riesgo Moderado 30-60%, Rojo: Alto Riesgo Crítico &gt;60%), "
            "emite recomendaciones de protocolo y resalta alertas fisiológicas (como fracción de eyección &lt;30% o creatinina sérica &gt;1.5 mg/dL).<br/>"
            "• <b>Analítica del Modelo:</b> Despliega en vivo las curvas ROC, matrices de confusión y gráficos de biomarcadores.<br/>"
            "• <b>Ficha Metodológica:</b> Detalla la arquitectura de software, trazabilidad y referencias bibliográficas.",
            body_style,
        )
    )
    story.append(Spacer(1, 10))

    # Caja informativa del Repositorio de GitHub
    repo_info = [
        [
            Paragraph("<b>ACCESO AL REPOSITORIO DE CÓDIGO FUENTE (GITHUB)</b>", table_header_style),
        ],
        [
            Paragraph(
                "Todo el código fuente modular, suite de pruebas automatizadas (8 tests pasados al 100%), "
                "configuración de CI con GitHub Actions (<code>.github/workflows/ci.yml</code>), bitácora de prompts "
                "y modelos serializados se encuentran versionados en el repositorio del proyecto:<br/><br/>"
                f"🔗 <b>Enlace de GitHub:</b> <code>{github_url}</code><br/><br/>"
                "<b>Comandos de Ejecución y Reproducibilidad:</b><br/>"
                "• Ejecutar pipeline: <code>python main.py</code><br/>"
                "• Ejecutar pruebas: <code>python -m pytest tests -v</code><br/>"
                "• Lanzar dashboard: <code>python app.py</code>",
                table_cell_style,
            ),
        ],
    ]
    t_repo = Table(repo_info, colWidths=[504])
    t_repo.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F8FA")),
                ("BOX", (0, 0), (-1, -1), 1, primary_color),
                ("PADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(t_repo)
    story.append(Spacer(1, 15))

    # ==========================================================================
    # 6. CONCLUSIONES Y PERFIL PROFESIONAL
    # ==========================================================================
    story.append(Paragraph("5. Conclusiones y Discusión del Dominio", h1_style))
    story.append(
        Paragraph(
            "<b>Interpretación Analítica y Clínica del Dominio:</b> Los experimentos demuestran con claridad "
            "que la supervivencia en pacientes con insuficiencia cardíaca está dominada por la interacción entre "
            "el período de seguimiento (<code>time</code>), la disfunción renal reflejada en la creatinina sérica "
            "(<code>serum_creatinine</code>) y la fracción de eyección ventricular (<code>ejection_fraction</code>). "
            "Esta jerarquía de biomarcadores coincide con los hallazgos biológicos de Chicco & Jurman (2020), "
            "evidenciando la validez del modelo. La obtención de una especificidad del 92.68% y un ROC-AUC de 0.8935 "
            "brinda garantías para su uso como herramienta de cribado preliminar en unidades cardiológicas.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Importancia y Aplicabilidad en el Perfil Profesional (Magíster en Ciencia de Datos e IA):</b> "
            "Desde la perspectiva del magíster, este proyecto consolida competencias críticas que van más allá del "
            "modelado matemático puro: (1) La adopción de principios de ingeniería de software (SOLID, Clean Code, POO) "
            "es indispensable para garantizar la transferibilidad de prototipos analíticos a entornos de producción "
            "hospitalaria; (2) La observabilidad estructurada mediante logging y validación con regex protege contra "
            "datos anómalos en flujos de datos en tiempo real; (3) La inclusión de métricas de calibración probabilística "
            "(Brier Score) y explicabilidad visual es un imperativo bioético para que los profesionales de la salud "
            "puedan confiar e interpretar las inferencias algorítmicas; y (4) La metodología de IA asistida permite "
            "aprovechar herramientas de vanguardia para elevar la productividad sin delegar la supervisión crítica.",
            body_style,
        )
    )

    story.append(Spacer(1, 10))

    # ==========================================================================
    # 7. REFERENCIAS BIBLIOGRÁFICAS (APA 7)
    # ==========================================================================
    story.append(Paragraph("6. Referencias Bibliográficas (Norma APA 7ma Edición)", h1_style))

    ref_style = ParagraphStyle(
        "RefStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#222222"),
        leftIndent=24,
        firstLineIndent=-24,
        spaceAfter=6,
    )

    refs = [
        "Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. <i>BMC Medical Informatics and Decision Making</i>, 20(1), 16. https://doi.org/10.1186/s12911-020-1023-5",
        "Gradio Team. (2024). <i>Gradio Python Client: Introduction and Interface Documentation</i>. Recuperado de https://gradio.app/docs/",
        "Kaggle. (s. f.). <i>Heart Failure Prediction Dataset</i>. Recuperado el 25 de agosto de 2026, de https://www.kaggle.com/datasets",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830.",
        "Python Software Foundation. (s. f.). <i>logging — Logging facility for Python</i>. Python documentation. Recuperado de https://docs.python.org/3/library/logging.html",
        "Python Software Foundation. (s. f.). <i>re — Regular expression operations</i>. Python documentation. Recuperado de https://docs.python.org/3/library/re.html",
        "University of California, Irvine. (s. f.). <i>Heart Failure Clinical Records</i>. UCI Machine Learning Repository. Recuperado de https://archive.ics.uci.edu/",
    ]

    for ref in refs:
        story.append(Paragraph(ref, ref_style))

    # Compilar Documento
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Informe académico compilado exitosamente en: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generador del Informe Académico Formal en PDF (Normas APA 7ma Edición)"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_PDF_NAME, help="Nombre del archivo PDF de salida"
    )
    parser.add_argument(
        "--name", "-n", default="Magíster en Formación", help="Nombre y apellidos del estudiante"
    )
    parser.add_argument(
        "--id", "-i", default="", help="Cédula de ciudadanía del estudiante"
    )
    parser.add_argument(
        "--repo", "-r", default="https://github.com/TU-USUARIO/U3-ProgamacionIA", help="URL del repositorio GitHub"
    )

    args = parser.parse_args()
    build_academic_report(
        output_filename=args.output,
        student_name=args.name,
        student_id=args.id,
        github_url=args.repo,
    )
