"""
Downloadable PDF report generation for a single prediction, using
ReportLab (pure Python, no external binary dependency — important for a
constrained Render deployment).
"""
from __future__ import annotations

import base64
import os
import uuid
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from model.predictor import PredictionResult
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0B5FA5"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeader",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0B5FA5"),
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Disclaimer",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#555555"),
        )
    )
    return styles


def generate_prediction_report(
    prediction: PredictionResult,
    original_image_path: str,
    reports_dir: str,
) -> str:
    """Render a single-prediction PDF report and return its file path."""
    os.makedirs(reports_dir, exist_ok=True)
    styles = _build_styles()
    output_path = os.path.join(reports_dir, f"report_{uuid.uuid4().hex}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
    )

    story = []
    story.append(Paragraph("RetinaAI — Prediction Report", styles["ReportTitle"]))
    story.append(
        Paragraph(
            f"Generated: {prediction.timestamp} &nbsp;|&nbsp; "
            f"Model: {prediction.architecture or 'N/A'} "
            f"v{prediction.model_version or 'N/A'}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    # --- Images side by side: original + Grad-CAM ---
    image_row = []
    if os.path.isfile(original_image_path):
        image_row.append(Image(original_image_path, width=7 * cm, height=7 * cm))
    if prediction.gradcam_base64:
        gradcam_bytes = base64.b64decode(prediction.gradcam_base64)
        image_row.append(Image(BytesIO(gradcam_bytes), width=7 * cm, height=7 * cm))

    if image_row:
        image_table = Table([image_row], colWidths=[7.5 * cm] * len(image_row))
        image_table.setStyle(
            TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])
        )
        story.append(image_table)
        captions = ["Uploaded Fundus Image"]
        if prediction.gradcam_base64:
            captions.append("Grad-CAM Attention Heatmap")
        caption_table = Table(
            [[Paragraph(c, styles["Italic"]) for c in captions]],
            colWidths=[7.5 * cm] * len(captions),
        )
        story.append(caption_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("Prediction Summary", styles["SectionHeader"]))

    summary_data = [
        ["Predicted Condition", prediction.predicted_class.replace("_", " ")],
        ["Confidence", f"{prediction.confidence * 100:.2f}%"],
        ["Risk Level", prediction.risk_level],
        ["Inference Time", f"{prediction.inference_time_ms:.1f} ms"],
    ]
    summary_table = Table(summary_data, colWidths=[5.5 * cm, 9.5 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2FB")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Probability Breakdown", styles["SectionHeader"]))
    prob_rows = [["Class", "Probability"]]
    for name, prob in sorted(
        prediction.all_probabilities.items(), key=lambda kv: kv[1], reverse=True
    ):
        prob_rows.append([name.replace("_", " "), f"{prob * 100:.2f}%"])
    prob_table = Table(prob_rows, colWidths=[9 * cm, 6 * cm])
    prob_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5FA5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ]
        )
    )
    story.append(prob_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Explanation", styles["SectionHeader"]))
    story.append(Paragraph(prediction.explanation, styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Medical Disclaimer", styles["SectionHeader"]))
    story.append(Paragraph(prediction.disclaimer, styles["Disclaimer"]))

    doc.build(story)
    logger.info("Generated PDF report at '%s'", output_path)
    return output_path
