from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from docx import Document
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_SAMPLE_DIR = ROOT / "tests" / "sample_data"


def _patient_text_block() -> str:
    return (
        "Patient Record\n"
        f"Patient ID: {uuid.uuid4()}\n"
        "Age: 65\n"
        "Gender: Male\n"
        "Symptoms: chest pain, moderate shortness of breath, dizziness\n"
        "BP: 175/98\n"
        "Heart Rate: 108\n"
        "Temperature: 100.8\n"
        "Medical History: diabetes, hypertension, heart disease\n"
    )


def create_pdf(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(file_path), pagesize=LETTER)
    text_object = c.beginText(50, 740)
    for line in _patient_text_block().splitlines():
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()


def create_docx(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    for line in _patient_text_block().splitlines():
        document.add_paragraph(line)
    document.save(str(file_path))


def create_batch_csv(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "Patient_ID": str(uuid.uuid4()),
            "Age": 68,
            "Gender": "Female",
            "Symptoms": "chest pain,palpitations",
            "Blood Pressure": "182/112",
            "Heart Rate": 126,
            "Temperature": 101.7,
            "Pre-Existing Conditions": "heart disease,hypertension",
        },
        {
            "Patient_ID": str(uuid.uuid4()),
            "Age": 24,
            "Gender": "Male",
            "Symptoms": "cold,sore throat",
            "Blood Pressure": "118/76",
            "Heart Rate": 74,
            "Temperature": 98.6,
            "Pre-Existing Conditions": "none",
        },
        {
            "Patient_ID": str(uuid.uuid4()),
            "Age": 14,
            "Gender": "Other",
            "Symptoms": "high fever,cough",
            "Blood Pressure": "122/79",
            "Heart Rate": 102,
            "Temperature": 102.4,
            "Pre-Existing Conditions": "asthma",
        },
    ]
    pd.DataFrame(rows).to_csv(file_path, index=False)


def main() -> None:
    create_pdf(DATA_DIR / "sample_ehr.pdf")
    create_pdf(TEST_SAMPLE_DIR / "test_ehr.pdf")
    create_docx(TEST_SAMPLE_DIR / "test_ehr.docx")
    create_batch_csv(TEST_SAMPLE_DIR / "test_batch.csv")
    print("Sample documents generated successfully")


if __name__ == "__main__":
    main()
