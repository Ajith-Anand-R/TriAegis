from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Dict, List

import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def single_result_pdf(patient: Dict[str, object], prediction: Dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)

    y = 760
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "AI Patient Triage Report")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated: {datetime.now(UTC).isoformat()}")
    y -= 20

    lines = [
        f"Patient ID: {patient.get('Patient_ID', '')}",
        f"Age/Gender: {patient.get('Age', '')} / {patient.get('Gender', '')}",
        f"Symptoms: {patient.get('Symptoms', '')}",
        f"Blood Pressure: {patient.get('Blood Pressure', '')}",
        f"Heart Rate: {patient.get('Heart Rate', '')}",
        f"Temperature: {patient.get('Temperature', '')}",
        f"Conditions: {patient.get('Pre-Existing Conditions', '')}",
        "",
        f"Risk Level: {prediction.get('risk_level', '')}",
        f"Confidence: {prediction.get('confidence', 0.0)}",
        f"Priority: {prediction.get('priority_score', '')} ({prediction.get('priority_category', '')})",
        f"Department: {prediction.get('department', '')}",
        f"Estimated Wait: {prediction.get('estimated_wait_time', '')}",
        f"Reason: {prediction.get('department_reason', '')}",
    ]

    for line in lines:
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = 760
        c.drawString(40, y, str(line)[:120])
        y -= 16

    c.save()
    buffer.seek(0)
    return buffer.read()


def batch_results_pdf(results: List[Dict[str, object]]) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)

    y = 760
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Batch Triage Summary")
    y -= 20

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated: {datetime.now(UTC).isoformat()}")
    y -= 20
    c.drawString(40, y, f"Total records: {len(results)}")
    y -= 20

    headers = "Priority  Patient ID                          Risk    Department"
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, headers)
    y -= 14
    c.setFont("Helvetica", 9)

    for row in sorted(results, key=lambda item: float(item.get("priority_score", 0)), reverse=True):
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = 760
        line = (
            f"{str(row.get('priority_score', '')):<8} "
            f"{str(row.get('patient_id', ''))[:34]:<34} "
            f"{str(row.get('risk_level', '')):<7} "
            f"{str(row.get('department', ''))[:22]}"
        )
        c.drawString(40, y, line)
        y -= 13

    c.save()
    buffer.seek(0)
    return buffer.read()


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")
