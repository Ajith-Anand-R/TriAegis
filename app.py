from __future__ import annotations

import io
import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.database import TriageDatabase
from utils.document_parser import parse_document
from utils.ehr_analyzer import analyze_ehr_history, parse_ehr_csv_from_bytes, EHRHistorySummary
from utils.explainer import ShapExplainer
from utils.healthcheck import healthcheck_dataframe, run_healthcheck
from utils.ml_engine import TriageMLEngine
from utils.reporting import batch_results_pdf, dataframe_to_csv_bytes, single_result_pdf
from utils.validators import validate_patient_payload


ROOT = Path(__file__).resolve().parent
SYMPTOM_OPTIONS = [
    "chest pain",
    "severe shortness of breath",
    "confusion",
    "severe bleeding",
    "loss of consciousness",
    "stroke symptoms",
    "severe abdominal pain",
    "difficulty breathing",
    "severe allergic reaction",
    "uncontrolled bleeding",
    "seizure",
    "severe trauma",
    "moderate shortness of breath",
    "high fever",
    "persistent vomiting",
    "severe headache",
    "palpitations",
    "moderate bleeding",
    "severe pain",
    "dizziness",
    "fainting",
    "dehydration",
    "abdominal pain",
    "irregular heartbeat",
    "mild headache",
    "cough",
    "cold",
    "minor pain",
    "fatigue",
    "nausea",
    "mild fever",
    "sore throat",
    "runny nose",
    "muscle ache",
    "rash",
    "minor injury",
    "constipation",
    "mild dizziness",
    "back pain",
    "joint pain",
    "insomnia",
    "anxiety",
    "minor cut",
    "sprain",
]

CONDITION_OPTIONS = [
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "COPD",
    "kidney disease",
    "obesity",
    "cancer",
    "stroke history",
    "arthritis",
    "high cholesterol",
    "thyroid disorder",
    "anxiety",
    "depression",
]

RISK_COLOR_MAP = {"Low": "#28A745", "Medium": "#FFA500", "High": "#DC3545"}


@st.cache_resource
def get_engine() -> TriageMLEngine:
    return TriageMLEngine(ROOT)


@st.cache_resource
def get_db() -> TriageDatabase:
    return TriageDatabase(ROOT / "database" / "patients.db")


@st.cache_resource
def get_explainer() -> ShapExplainer:
    return ShapExplainer.from_project_root(ROOT)


def _risk_color(risk: str) -> str:
    if risk == "High":
        return "#DC3545"
    if risk == "Medium":
        return "#FFA500"
    return "#28A745"


def _priority_color(category: str) -> str:
    mapping = {
        "Critical": "#DC3545",
        "Urgent": "#ff6f00",
        "High": "#f6b300",
        "Standard": "#1f77b4",
        "Low": "#28A745",
    }
    return mapping.get(category, "#1f77b4")


def _section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def _style_plot(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=24, r=24, t=56, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        legend_title_text="",
    )
    return fig


def _normalize_date_range(start_date: date, end_date: date) -> tuple[date, date, bool]:
    if end_date < start_date:
        return end_date, start_date, True
    return start_date, end_date, False


def _pretty_label(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def _prettify_table(df: pd.DataFrame) -> pd.DataFrame:
    pretty_df = df.copy()
    if "status" in pretty_df.columns:
        pretty_df["status"] = pretty_df["status"].astype(str).map(_pretty_label)
    if "source" in pretty_df.columns:
        pretty_df["source"] = pretty_df["source"].astype(str).map(_pretty_label)
    pretty_df = pretty_df.rename(columns={column: _pretty_label(column) for column in pretty_df.columns})
    return pretty_df


def _build_patient_payload(
    patient_id: str,
    age: int,
    gender: str,
    symptoms: List[str],
    systolic: int,
    diastolic: int,
    heart_rate: int,
    temperature: float,
    conditions: List[str],
) -> Dict[str, object]:
    return {
        "Patient_ID": patient_id,
        "Age": int(age),
        "Gender": gender,
        "Symptoms": ",".join(symptoms),
        "Blood Pressure": f"{systolic}/{diastolic}",
        "Heart Rate": int(heart_rate),
        "Temperature": float(temperature),
        "Pre-Existing Conditions": ",".join(conditions) if conditions else "none",
    }


def render_single_patient_tab() -> None:
    _section_header("Single Patient Analysis", "Capture patient context and run triage.")
    col1, col2, col3 = st.columns([0.40, 0.35, 0.25])

    with col1:
        input_card = st.container(border=True)
        with input_card:
            st.markdown("### Patient Input")
            st.caption("Enter demographics, symptoms, and vitals.")
            left_a, left_b = st.columns([0.75, 0.25])
            with left_a:
                patient_id = st.text_input(
                    "Patient ID",
                    value=str(uuid.uuid4()),
                    placeholder="e.g. PAT-2026-0001",
                    help="Use a unique identifier to keep the patient timeline accurate.",
                )
            with left_b:
                if st.button("Generate UUID"):
                    st.rerun()

            age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            symptoms = st.multiselect(
                "Symptoms",
                SYMPTOM_OPTIONS,
                help="Select one or more presenting symptoms.",
            )

            st.markdown("#### Vital Signs")
            bp_c1, bp_c2 = st.columns(2)
            with bp_c1:
                systolic = st.number_input("BP Systolic", min_value=70, max_value=260, value=120)
            with bp_c2:
                diastolic = st.number_input("BP Diastolic", min_value=40, max_value=160, value=80)

            heart_rate = st.number_input("Heart Rate", min_value=30, max_value=220, value=70)
            temperature = st.number_input("Temperature (°F)", min_value=90.0, max_value=112.0, value=98.6, step=0.1)

            conditions = st.multiselect(
                "Pre-Existing Conditions",
                CONDITION_OPTIONS,
                help="Optional conditions that may influence triage risk.",
            )

            uploaded = st.file_uploader(
                "Upload Medical Document",
                type=["pdf", "docx", "csv"],
                help="Optional: extract structured details from a document upload.",
            )
            if uploaded is not None:
                temp_path = ROOT / "data" / f"tmp_upload_{uploaded.name}"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_bytes(uploaded.read())
                try:
                    parsed = parse_document(temp_path)
                    with st.expander("Extracted Data", expanded=True):
                        st.write(parsed[:3])
                except Exception as exc:
                    st.error(f"Could not parse the uploaded document: {exc}")
                finally:
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)

            st.markdown("#### 📋 EHR / EMR History")
            st.caption("Upload the patient's Electronic Health Record to factor medical history into risk analysis.")
            ehr_file = st.file_uploader(
                "Upload EHR/EMR CSV",
                type=["csv"],
                key="ehr_upload",
                help="CSV with columns: Date, Record_Type, Description, Vitals, Labs, Medications, etc.",
            )

            ehr_score = 0.0
            ehr_summary = None
            if ehr_file is not None:
                try:
                    raw_bytes = ehr_file.read()
                    ehr_records = parse_ehr_csv_from_bytes(raw_bytes, filename=ehr_file.name)
                    ehr_summary = analyze_ehr_history(ehr_records, filename=ehr_file.name)
                    ehr_score = ehr_summary.history_risk_score
                    st.session_state["ehr_summary"] = ehr_summary

                    with st.expander("EHR History Summary", expanded=True):
                        ehr_cols = st.columns(4)
                        ehr_cols[0].metric("Total Records", ehr_summary.total_records)
                        ehr_cols[1].metric("Visits", ehr_summary.total_visits)
                        ehr_cols[2].metric("Hospitalizations", ehr_summary.hospitalizations)
                        ehr_cols[3].metric("ER Visits", ehr_summary.er_visits)

                        ehr_cols2 = st.columns(4)
                        ehr_cols2[0].metric("Chronic Conditions", len(ehr_summary.chronic_conditions))
                        ehr_cols2[1].metric("Active Meds", ehr_summary.active_medications)
                        ehr_cols2[2].metric("Abnormal Labs", f"{ehr_summary.abnormal_labs}/{ehr_summary.total_labs}")
                        ehr_cols2[3].metric("Procedures", ehr_summary.procedures)

                        score_color = "#e74c3c" if ehr_score >= 2.0 else ("#f39c12" if ehr_score >= 1.0 else "#27ae60")
                        st.markdown(
                            f"<div style='padding:10px;border-radius:8px;background:{score_color};color:white;"
                            f"text-align:center;font-weight:700;font-size:1.1rem;'>"
                            f"📊 EHR History Risk Score: {ehr_score:.2f} / 3.00</div>",
                            unsafe_allow_html=True,
                        )

                        if ehr_summary.chronic_conditions:
                            st.markdown("**Chronic Conditions:** " + ", ".join(ehr_summary.chronic_conditions))

                        if ehr_summary.risk_factors:
                            st.markdown("**Risk Factors:**")
                            for rf in ehr_summary.risk_factors:
                                st.markdown(f"- {rf}")

                        breakdown = ehr_summary.score_breakdown
                        if breakdown:
                            bd_df = pd.DataFrame(
                                {"Component": [k.replace("_", " ").title() for k in breakdown],
                                 "Points": list(breakdown.values())}
                            )
                            fig_bd = px.bar(
                                bd_df, x="Points", y="Component", orientation="h",
                                title="Score Breakdown",
                                color="Points",
                                color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
                            )
                            fig_bd.update_layout(height=220, margin=dict(l=8, r=8, t=40, b=8))
                            st.plotly_chart(fig_bd, use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not parse EHR file: {exc}")
                    ehr_score = 0.0

            patient_payload = _build_patient_payload(
                patient_id=patient_id,
                age=age,
                gender=gender,
                symptoms=symptoms,
                systolic=systolic,
                diastolic=diastolic,
                heart_rate=heart_rate,
                temperature=temperature,
                conditions=conditions,
            )

            valid, errors = validate_patient_payload(patient_payload)
            analyze = st.button("Analyze Patient", type="primary", width="stretch", help="Run triage prediction")

    if "single_result" not in st.session_state:
        st.session_state["single_result"] = None
        st.session_state["single_explain"] = None

    if analyze:
        if not valid:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Running triage analysis..."):
                engine = get_engine()
                result = engine.predict_one(patient_payload, ehr_history_score=ehr_score)
                explainer = get_explainer()
                explanation = explainer.explain(patient_payload)
                st.session_state["single_result"] = {"patient": patient_payload, "prediction": result}
                st.session_state["single_explain"] = explanation

    single_result = st.session_state.get("single_result")
    single_explain = st.session_state.get("single_explain")

    with col2:
        result_card = st.container(border=True)
        with result_card:
            st.markdown("### Results")
            if single_result:
                pred = single_result["prediction"]
                risk_color = _risk_color(pred["risk_level"])
                st.markdown(
                    f"<div style='padding:12px;border-radius:10px;background:{risk_color};color:white;font-weight:700;'>"
                    f"{pred['risk_level']} RISK"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.metric("Priority Score", f"{pred['priority_score']}/10")
                st.metric("Priority Category", pred["priority_category"])
                st.metric("Queue Position", pred["queue_position"])
                st.metric("Estimated Wait", pred["estimated_wait_time"])
                st.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")
                ehr_hist_score = pred.get("ehr_history_score", 0.0)
                if ehr_hist_score > 0:
                    st.metric("EHR History Score", f"{ehr_hist_score:.2f} / 3.00")

                probs_df = pd.DataFrame(
                    {
                        "Risk": ["Low", "Medium", "High"],
                        "Probability": [
                            pred["probabilities"]["Low"],
                            pred["probabilities"]["Medium"],
                            pred["probabilities"]["High"],
                        ],
                    }
                )
                fig = px.bar(
                    probs_df,
                    x="Probability",
                    y="Risk",
                    orientation="h",
                    color="Risk",
                    color_discrete_map=RISK_COLOR_MAP,
                    title="Risk Probabilities",
                    labels={"Probability": "Probability", "Risk": "Risk Level"},
                )
                st.plotly_chart(_style_plot(fig), width="stretch")

                st.success(f"Recommended Department: {pred['department']}")
                st.caption(pred["department_reason"])

                save_db = st.button("Save to Database", width="stretch")
                if save_db:
                    db = get_db()
                    shap_rows = single_explain["explanation"]["top_contributors"] if single_explain else []
                    db.save_prediction(
                        single_result["patient"],
                        single_result["prediction"],
                        shap_top_contributors=shap_rows,
                        source="manual",
                    )
                    st.success("Result saved to patient history")

                if st.button("Clear Current Result", width="stretch"):
                    st.session_state["single_result"] = None
                    st.session_state["single_explain"] = None
                    st.rerun()

                single_pdf_bytes = single_result_pdf(single_result["patient"], single_result["prediction"])
                st.download_button(
                    "Export as PDF",
                    data=single_pdf_bytes,
                    file_name=f"triage_{single_result['prediction']['patient_id']}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            else:
                st.info("Run an analysis to see triage results")

    with col3:
        explain_card = st.container(border=True)
        with explain_card:
            st.markdown("### Why This Prediction?")
            if single_explain:
                contrib_df = pd.DataFrame(single_explain["explanation"]["top_contributors"])
                if not contrib_df.empty:
                    chart = px.bar(
                        contrib_df,
                        x="impact",
                        y="feature",
                        orientation="h",
                        title="Top SHAP Contributors",
                        color="impact",
                        color_continuous_scale=["#28A745", "#DC3545"],
                        labels={"impact": "Impact", "feature": "Feature"},
                    )
                    st.plotly_chart(_style_plot(chart), width="stretch")
                    st.dataframe(
                        contrib_df[["feature", "value", "impact", "interpretation"]],
                        width="stretch",
                        hide_index=True,
                    )
                for factor in single_explain["explanation"]["confidence_factors"][:3]:
                    st.caption(f"• {factor}")
            else:
                st.info("Explainability details appear after analysis")


def render_batch_tab() -> None:
    _section_header("Batch Processing", "Upload CSV and process triage in bulk.")
    uploaded = st.file_uploader("Upload CSV with patient data", type=["csv"], key="batch_csv")

    if uploaded is None:
        st.info("Upload a CSV file to begin batch triage")
        return

    temp_path = ROOT / "data" / f"tmp_batch_{uploaded.name}"
    temp_path.write_bytes(uploaded.read())

    try:
        rows = parse_document(temp_path)
        preview = pd.DataFrame(rows)
        with st.container(border=True):
            st.markdown("#### Source Preview")
            st.dataframe(preview.head(10), width="stretch", hide_index=True)
            st.success(f"Loaded {len(preview)} records")
            st.caption("Preview: first 10 rows.")

        if st.button("Process All Patients", type="primary"):
            engine = get_engine()
            with st.spinner("Processing patient batch..."):
                results = engine.predict_batch(rows)
            result_df = pd.DataFrame(results)
            st.session_state["batch_results"] = result_df

        result_df = st.session_state.get("batch_results")
        if result_df is not None and not result_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(result_df))
            c2.metric("High Risk", int((result_df["risk_level"] == "High").sum()))
            c3.metric("Medium Risk", int((result_df["risk_level"] == "Medium").sum()))
            c4.metric("Low Risk", int((result_df["risk_level"] == "Low").sum()))

            with st.container(border=True):
                st.markdown("#### Priority Queue Preview")
                queue_df = result_df.sort_values(["priority_score", "queue_position"], ascending=[False, True])
                st.dataframe(_prettify_table(queue_df), width="stretch", hide_index=True)

            action_a, action_b, action_c = st.columns(3)
            with action_a:
                save_all = st.button("Save All to Database", width="stretch")
            with action_b:
                csv_bytes = dataframe_to_csv_bytes(result_df)
                st.download_button(
                    "Download Results CSV",
                    data=csv_bytes,
                    file_name="batch_results.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with action_c:
                pdf_bytes = batch_results_pdf(result_df.to_dict(orient="records"))
                st.download_button(
                    "Download Priority Queue PDF",
                    data=pdf_bytes,
                    file_name="priority_queue_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

            if save_all:
                db = get_db()
                for index, row in result_df.iterrows():
                    patient = rows[index] if index < len(rows) else rows[-1]
                    db.save_prediction(patient, row.to_dict(), source="batch")
                st.success("Batch results saved to patient history")

    except Exception as exc:
        st.error(f"Batch processing could not be completed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)


def render_queue_tab() -> None:
    _section_header("Priority Queue Management", "Manage queue load and urgent cases.")
    db = get_db()
    controls_col, action_col = st.columns([0.62, 0.38])
    status_options = {
        "All": "all",
        "Waiting": "waiting",
        "In Progress": "in_progress",
        "Completed": "completed",
    }
    with controls_col:
        status_label = st.selectbox("Queue Status", list(status_options.keys()), index=0)
    status_filter = status_options[status_label]
    queue_data = db.get_priority_queue(status=None if status_filter == "all" else status_filter)
    queue = pd.DataFrame(queue_data)
    if queue.empty:
        st.info("No queue records found for the selected status")
        return

    with controls_col:
        dept_filter = st.multiselect(
            "Department Filter",
            options=sorted(queue["department"].dropna().unique().tolist()),
            default=[],
        )
    if dept_filter:
        queue = queue[queue["department"].isin(dept_filter)]

    with st.container(border=True):
        display_queue = _prettify_table(queue)
        st.dataframe(display_queue, width="stretch", hide_index=True)

    waiting = queue[queue["status"] == "waiting"]

    dept_load = pd.DataFrame(db.get_department_load())
    if not dept_load.empty:
        overload_departments = dept_load[dept_load["waiting_count"] >= 10]
        critical_waiting = waiting[waiting["priority_score"] >= 9.0]
        if not critical_waiting.empty:
            st.warning(f"Priority alert: {len(critical_waiting)} critical patients are waiting")
        if not overload_departments.empty:
            st.warning(
                "Department overload detected: " + ", ".join(overload_departments["department"].tolist())
            )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients Waiting", len(waiting))
    c2.metric("Critical Patients", int((waiting["priority_score"] >= 9.0).sum()))
    c3.metric("Avg Wait", "-" if waiting.empty else waiting["estimated_wait_time"].iloc[0])
    c4.metric("Longest Queue Position", int(waiting["queue_position"].max()) if not waiting.empty else 0)

    with action_col:
        st.markdown("#### Queue Actions")
        if st.button("Call Next Patient", width="stretch", disabled=waiting.empty):
            top = waiting.sort_values("queue_position").head(1)
            if not top.empty:
                db.update_queue_status(int(top.iloc[0]["queue_id"]), "in_progress")
                st.success("Top queue patient moved to in progress")
                st.rerun()

        mark_completed_id = st.number_input("Queue ID to mark completed", min_value=0, step=1)
        if st.button("Mark as Completed", width="stretch") and mark_completed_id > 0:
            db.update_queue_status(int(mark_completed_id), "completed")
            st.success(f"Queue item {int(mark_completed_id)} marked completed")
            st.rerun()

        if st.button("Clear Completed (>24h)", width="stretch"):
            deleted = db.clear_completed()
            st.success(f"Removed {deleted} completed queue records")
            st.rerun()


def render_analytics_tab() -> None:
    _section_header("Analytics Dashboard", "Track performance, risk, and department trends.")
    db = get_db()
    queue = pd.DataFrame(db.get_priority_queue())
    high = pd.DataFrame(db.get_high_risk_patients(limit=500))
    predictions = pd.DataFrame(db.get_predictions(limit=5000))

    if predictions.empty and queue.empty:
        st.info("No analytics data available yet")
        return

    if not predictions.empty:
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
        predictions = predictions.dropna(subset=["timestamp"])
        predictions["date"] = predictions["timestamp"].dt.date

    st.markdown("#### Filters")
    filter_container = st.container(border=True)
    f1, f2, f3, f4 = filter_container.columns([0.28, 0.28, 0.28, 0.16])

    if predictions.empty:
        start_date = date.today() - timedelta(days=30)
        end_date = date.today()
        risk_filter: List[str] = []
        dept_filter: List[str] = []
    else:
        min_date = predictions["date"].min()
        max_date = predictions["date"].max()
        with f1:
            start_date = st.date_input(
                "Start Date",
                value=max(min_date, max_date - timedelta(days=30)),
                key="analytics_start_date",
            )
        with f2:
            end_date = st.date_input("End Date", value=max_date, key="analytics_end_date")
        risk_options = sorted(predictions["risk_level"].dropna().unique().tolist()) if "risk_level" in predictions.columns else []
        dept_options = (
            sorted(predictions["recommended_department"].dropna().unique().tolist())
            if "recommended_department" in predictions.columns
            else []
        )
        with f3:
            risk_filter = st.multiselect("Risk Level", risk_options, default=[], key="analytics_risk_level")
        dept_filter = filter_container.multiselect("Department", dept_options, default=[], key="analytics_department")
        with f4:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("Reset Filters", key="analytics_reset_filters", width="stretch"):
                for session_key in [
                    "analytics_start_date",
                    "analytics_end_date",
                    "analytics_risk_level",
                    "analytics_department",
                ]:
                    st.session_state.pop(session_key, None)
                st.rerun()

    start_date, end_date, swapped = _normalize_date_range(start_date, end_date)
    if swapped:
        st.info("Date range auto-corrected because End Date was earlier than Start Date")

    filtered_predictions = predictions.copy()
    if not filtered_predictions.empty:
        filtered_predictions = filtered_predictions[
            (filtered_predictions["date"] >= start_date) & (filtered_predictions["date"] <= end_date)
        ]
        if risk_filter:
            filtered_predictions = filtered_predictions[filtered_predictions["risk_level"].isin(risk_filter)]
        if dept_filter:
            filtered_predictions = filtered_predictions[
                filtered_predictions["recommended_department"].isin(dept_filter)
            ]

    waiting_queue = queue.copy()
    if "status" in waiting_queue.columns:
        waiting_queue = waiting_queue[waiting_queue["status"] == "waiting"]

    total_predictions = len(filtered_predictions)
    high_risk_rate = (
        float((filtered_predictions["risk_level"] == "High").mean() * 100)
        if total_predictions > 0 and "risk_level" in filtered_predictions.columns
        else 0.0
    )
    avg_priority = (
        float(filtered_predictions["priority_score"].mean())
        if total_predictions > 0 and "priority_score" in filtered_predictions.columns
        else 0.0
    )
    avg_confidence = (
        float(filtered_predictions["model_confidence"].mean() * 100)
        if total_predictions > 0 and "model_confidence" in filtered_predictions.columns
        else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", total_predictions)
    c2.metric("High-Risk Share", f"{high_risk_rate:.1f}%")
    c3.metric("Average Priority", f"{avg_priority:.2f}")
    c4.metric("Average Confidence", f"{avg_confidence:.1f}%")

    q1, q2, q3 = st.columns(3)
    q1.metric("Queue (Waiting)", int(len(waiting_queue)))
    q2.metric(
        "Critical Waiting",
        int((waiting_queue["priority_score"] >= 9.0).sum())
        if "priority_score" in waiting_queue.columns
        else 0,
    )
    q3.metric("Recent High-Risk (500)", int(len(high)))
    st.caption(
        f"Date: {start_date} → {end_date} | "
        f"Risk: {', '.join(risk_filter) if risk_filter else 'All'} | "
        f"Dept: {', '.join(dept_filter) if dept_filter else 'All'}"
    )

    if total_predictions == 0:
        st.warning("No records match the current filter combination")
        return

    left_top, right_top = st.columns(2)
    with left_top:
        if "risk_level" in filtered_predictions.columns:
            risk_counts = filtered_predictions["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            fig_risk_mix = px.pie(
                risk_counts,
                names="risk_level",
                values="count",
                title="Risk Mix",
                color="risk_level",
                color_discrete_map=RISK_COLOR_MAP,
                labels={"risk_level": "Risk Level", "count": "Cases"},
            )
            st.plotly_chart(_style_plot(fig_risk_mix), width="stretch")
    with right_top:
        if "recommended_department" in filtered_predictions.columns:
            dept_counts = (
                filtered_predictions["recommended_department"].value_counts().head(10).reset_index()
            )
            dept_counts.columns = ["department", "count"]
            fig_dept_load = px.bar(
                dept_counts,
                x="department",
                y="count",
                title="Top Departments by Volume",
                labels={"department": "Department", "count": "Cases"},
            )
            st.plotly_chart(_style_plot(fig_dept_load), width="stretch")

    left_mid, right_mid = st.columns(2)
    with left_mid:
        trend = (
            filtered_predictions.groupby(["date", "risk_level"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            if "risk_level" in filtered_predictions.columns
            else filtered_predictions.groupby(["date"], as_index=False).size().rename(columns={"size": "count"})
        )
        fig_trend = px.line(
            trend,
            x="date",
            y="count",
            color="risk_level" if "risk_level" in trend.columns else None,
            title="Daily Triage Trend",
            labels={"date": "Date", "count": "Cases", "risk_level": "Risk Level"},
        )
        st.plotly_chart(_style_plot(fig_trend), width="stretch")
    with right_mid:
        if {"recommended_department", "priority_score"}.issubset(filtered_predictions.columns):
            fig_priority_box = px.box(
                filtered_predictions,
                x="recommended_department",
                y="priority_score",
                title="Priority Spread by Department",
                labels={"recommended_department": "Department", "priority_score": "Priority Score"},
            )
            st.plotly_chart(_style_plot(fig_priority_box), width="stretch")

    left_bottom, right_bottom = st.columns(2)
    with left_bottom:
        if "priority_category" in filtered_predictions.columns:
            category_counts = filtered_predictions["priority_category"].value_counts().reset_index()
            category_counts.columns = ["priority_category", "count"]
            fig_priority_cat = px.bar(
                category_counts,
                x="priority_category",
                y="count",
                title="Priority Category Distribution",
                labels={"priority_category": "Priority Category", "count": "Cases"},
            )
            st.plotly_chart(_style_plot(fig_priority_cat), width="stretch")
    with right_bottom:
        if {"risk_level", "model_confidence"}.issubset(filtered_predictions.columns):
            fig_conf = px.box(
                filtered_predictions,
                x="risk_level",
                y="model_confidence",
                color="risk_level",
                color_discrete_map=RISK_COLOR_MAP,
                title="Model Confidence by Risk Level",
                labels={"risk_level": "Risk Level", "model_confidence": "Model Confidence"},
            )
            st.plotly_chart(_style_plot(fig_conf), width="stretch")

    if "symptoms" in filtered_predictions.columns:
        symptom_counter = (
            filtered_predictions["symptoms"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(12)
            .reset_index()
        )
        symptom_counter.columns = ["symptom", "count"]
        if not symptom_counter.empty:
            fig_symptoms = px.bar(
                symptom_counter,
                x="symptom",
                y="count",
                title="Top Symptoms",
                labels={"symptom": "Symptom", "count": "Cases"},
            )
            st.plotly_chart(_style_plot(fig_symptoms), width="stretch")

    st.markdown("#### Recent Activity")
    activity_cols = [
        "timestamp",
        "patient_id",
        "risk_level",
        "priority_score",
        "priority_category",
        "recommended_department",
        "model_confidence",
    ]
    available_cols = [column for column in activity_cols if column in filtered_predictions.columns]
    activity_view = filtered_predictions.sort_values("timestamp", ascending=False)[available_cols].head(30)
    st.dataframe(
        _prettify_table(activity_view),
        width="stretch",
        hide_index=True,
    )


def render_history_tab() -> None:
    _section_header("Patient History", "Filter and export triage records.")
    db = get_db()
    all_predictions = pd.DataFrame(db.get_predictions(limit=5000))

    if all_predictions.empty:
        st.info("No patient history records available")
        return

    filter_col_a, filter_col_b, filter_col_c = st.columns(3)
    with filter_col_a:
        search_query = st.text_input(
            "Search Patient ID",
            key="history_search_patient_id",
            placeholder="Enter patient ID or partial match",
        )
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30), key="history_start_date")
    with filter_col_b:
        end_date = st.date_input("End Date", value=date.today(), key="history_end_date")
        risk_filter = st.multiselect("Risk Level", ["Low", "Medium", "High"], default=[], key="history_risk_level")
    with filter_col_c:
        dept_options = sorted(all_predictions["recommended_department"].dropna().unique().tolist())
        dept_filter = st.multiselect("Department", dept_options, default=[], key="history_department")
        priority_filter = st.multiselect(
            "Priority Category",
            ["Critical", "Urgent", "High", "Standard", "Low"],
            default=[],
            key="history_priority_category",
        )
        if st.button("Reset Filters", key="history_reset_filters", width="stretch"):
            for session_key in [
                "history_search_patient_id",
                "history_start_date",
                "history_end_date",
                "history_risk_level",
                "history_department",
                "history_priority_category",
            ]:
                st.session_state.pop(session_key, None)
            st.rerun()

    start_date, end_date, swapped = _normalize_date_range(start_date, end_date)
    if swapped:
        st.info("Date range auto-corrected because End Date was earlier than Start Date")

    history_records = db.search_predictions(
        patient_id_query=search_query or None,
        risk_levels=risk_filter or None,
        departments=dept_filter or None,
        priority_categories=priority_filter or None,
        start_date=str(start_date),
        end_date=str(end_date),
        limit=5000,
    )
    history = pd.DataFrame(history_records)

    if history.empty:
        st.warning("No records match the current filter combination")
        return

    st.dataframe(_prettify_table(history), width="stretch", hide_index=True)
    action_left, action_mid, action_right = st.columns(3)
    csv_data = dataframe_to_csv_bytes(history)
    with action_left:
        st.download_button(
            "Export History (CSV)",
            data=csv_data,
            file_name="history_filtered.csv",
            mime="text/csv",
            width="stretch",
        )

    history_pdf = batch_results_pdf(
        history[["patient_id", "risk_level", "recommended_department", "priority_score"]]
        .rename(columns={"recommended_department": "department"})
        .to_dict(orient="records")
    )
    with action_mid:
        st.download_button(
            "Generate Statistics Report (PDF)",
            data=history_pdf,
            file_name="history_report_filtered.pdf",
            mime="application/pdf",
            width="stretch",
        )

    with action_right:
        if st.button("Clear Old Records (>90 days)", width="stretch"):
            deleted = db.clear_old_records(days=90)
            st.success(
                f"Deleted {deleted['deleted_prediction_rows']} prediction rows and "
                f"{deleted['deleted_queue_rows']} queue rows older than 90 days"
            )


def render_model_performance_tab() -> None:
    _section_header("Model Performance", "Review model metrics and training artifacts.")
    saved_models = ROOT / "models" / "saved_models"
    metrics_path = saved_models / "metrics.json"

    if not metrics_path.exists():
        st.warning("metrics.json not found. Train the model to generate performance artifacts.")
        return

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not read metrics.json: {exc}")
        return

    top_row = st.columns(4)
    top_row[0].metric("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.4f}")
    top_row[1].metric("ROC AUC (OvR)", f"{float(metrics.get('roc_auc_ovr', 0.0)):.4f}")
    top_row[2].metric("CV Mean", f"{float(metrics.get('cv_accuracy_mean', 0.0)):.4f}")
    top_row[3].metric("CV Std", f"{float(metrics.get('cv_accuracy_std', 0.0)):.4f}")

    per_class = metrics.get("per_class", {})
    if isinstance(per_class, dict) and per_class:
        per_class_rows = []
        for label in ["Low", "Medium", "High"]:
            values = per_class.get(label, {}) if isinstance(per_class.get(label), dict) else {}
            per_class_rows.append(
                {
                    "class": label,
                    "precision": float(values.get("precision", 0.0)),
                    "recall": float(values.get("recall", 0.0)),
                    "f1": float(values.get("f1", 0.0)),
                    "support": int(values.get("support", 0)),
                }
            )
        st.markdown("#### Per-Class Precision / Recall / F1")
        st.dataframe(pd.DataFrame(per_class_rows), width="stretch", hide_index=True)

    chart_specs = [
        (saved_models / "confusion_matrix.png", "Confusion Matrix"),
        (saved_models / "roc_curve.png", "ROC Curve (One-vs-Rest)"),
        (saved_models / "feature_importance.png", "Feature Importance"),
    ]
    for chart_path, caption in chart_specs:
        if chart_path.exists():
            st.image(str(chart_path), caption=caption, use_container_width=True)
        else:
            st.info(f"{caption} image not found at: {chart_path.name}")

    top_features = metrics.get("top_feature_importance", [])
    if isinstance(top_features, list) and top_features:
        st.markdown("#### Top Feature Importance")
        feat_df = pd.DataFrame(top_features)
        display_cols = [column for column in ["feature", "importance"] if column in feat_df.columns]
        if display_cols:
            st.dataframe(feat_df[display_cols], width="stretch", hide_index=True)


def _age_band(age: int) -> str:
    if age < 18:
        return "0-17"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    if age < 75:
        return "60-74"
    return "75+"


def _positive_rate(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = frame.groupby(group_col, as_index=False).agg(
        positive_rate=("predicted_high", "mean"),
        sample_size=("predicted_high", "count"),
    )
    grouped["positive_rate"] = grouped["positive_rate"].fillna(0.0)
    return grouped


def _equal_opportunity_rate(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    positive_truth = frame[frame["actual_high"] == 1]
    if positive_truth.empty:
        return pd.DataFrame(columns=[group_col, "tpr", "sample_size"])
    grouped = positive_truth.groupby(group_col, as_index=False).agg(
        tpr=("predicted_high", "mean"),
        sample_size=("predicted_high", "count"),
    )
    grouped["tpr"] = grouped["tpr"].fillna(0.0)
    return grouped


def _parity_diff(metric_df: pd.DataFrame, metric_col: str) -> float:
    if metric_df.empty:
        return 0.0
    return float(metric_df[metric_col].max() - metric_df[metric_col].min())


def render_fairness_tab() -> None:
    _section_header("Bias & Fairness Analysis", "Compare prediction behavior across age and gender cohorts.")

    dataset_path = ROOT / "data" / "synthetic_patients.csv"
    if not dataset_path.exists():
        st.warning("Dataset not found. Generate or restore data/synthetic_patients.csv first.")
        return

    source_df = pd.read_csv(dataset_path)
    required_cols = {
        "Patient_ID",
        "Age",
        "Gender",
        "Symptoms",
        "Blood Pressure",
        "Heart Rate",
        "Temperature",
        "Pre-Existing Conditions",
        "Risk_Level",
    }
    if not required_cols.issubset(set(source_df.columns)):
        missing = sorted(required_cols.difference(source_df.columns))
        st.error(f"Dataset missing required fairness columns: {', '.join(missing)}")
        return

    left, right = st.columns([0.6, 0.4])
    with left:
        sample_size = st.slider(
            "Evaluation sample size",
            min_value=300,
            max_value=min(5000, len(source_df)),
            value=min(2000, len(source_df)),
            step=100,
        )
    with right:
        run_eval = st.button("Run Fairness Analysis", type="primary", width="stretch")

    if not run_eval:
        st.info("Choose sample size and click Run Fairness Analysis.")
        return

    with st.spinner("Computing predictions and fairness metrics..."):
        eval_df = source_df.head(sample_size).copy()
        engine = get_engine()
        prediction_rows = engine.predict_batch(eval_df.to_dict(orient="records"))
        pred_map = {row["patient_id"]: row["risk_level"] for row in prediction_rows}

        eval_df["predicted_risk"] = eval_df["Patient_ID"].astype(str).map(pred_map)
        eval_df["age_band"] = eval_df["Age"].astype(int).apply(_age_band)
        eval_df["predicted_high"] = (eval_df["predicted_risk"] == "High").astype(int)
        eval_df["actual_high"] = (eval_df["Risk_Level"] == "High").astype(int)

        gender_dp = _positive_rate(eval_df, "Gender")
        age_dp = _positive_rate(eval_df, "age_band")
        gender_eo = _equal_opportunity_rate(eval_df, "Gender")
        age_eo = _equal_opportunity_rate(eval_df, "age_band")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Rows Analyzed", len(eval_df))
    g2.metric("Demographic Parity (Gender)", f"{_parity_diff(gender_dp, 'positive_rate'):.3f}")
    g3.metric("Demographic Parity (Age)", f"{_parity_diff(age_dp, 'positive_rate'):.3f}")
    g4.metric("Equal Opportunity (Gender)", f"{_parity_diff(gender_eo, 'tpr'):.3f}")
    st.caption(f"Equal Opportunity (Age): {_parity_diff(age_eo, 'tpr'):.3f}")

    dist_gender = (
        eval_df.groupby(["Gender", "predicted_risk"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    dist_age = (
        eval_df.groupby(["age_band", "predicted_risk"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_gender = px.bar(
            dist_gender,
            x="Gender",
            y="count",
            color="predicted_risk",
            barmode="group",
            color_discrete_map=RISK_COLOR_MAP,
            title="Prediction Distribution by Gender",
        )
        st.plotly_chart(_style_plot(fig_gender), width="stretch")
    with c2:
        fig_age = px.bar(
            dist_age,
            x="age_band",
            y="count",
            color="predicted_risk",
            barmode="group",
            color_discrete_map=RISK_COLOR_MAP,
            title="Prediction Distribution by Age Band",
        )
        st.plotly_chart(_style_plot(fig_age), width="stretch")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Demographic Parity Tables")
        st.dataframe(gender_dp.rename(columns={"positive_rate": "high_risk_rate"}), width="stretch", hide_index=True)
        st.dataframe(age_dp.rename(columns={"positive_rate": "high_risk_rate"}), width="stretch", hide_index=True)
    with t2:
        st.markdown("#### Equal Opportunity Tables")
        if gender_eo.empty:
            st.info("No positive ground-truth samples available for Equal Opportunity by Gender.")
        else:
            st.dataframe(gender_eo, width="stretch", hide_index=True)
        if age_eo.empty:
            st.info("No positive ground-truth samples available for Equal Opportunity by Age.")
        else:
            st.dataframe(age_eo, width="stretch", hide_index=True)


def _sim_default_state() -> Dict[str, object]:
    return {
        "current_minute": 0,
        "queue": [],
        "timeline": [],
        "completed": 0,
        "arrived": 0,
    }


def _sort_sim_queue(queue_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    sorted_rows = sorted(
        queue_rows,
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["age"]),
            int(row["arrival_minute"]),
        ),
    )
    for idx, row in enumerate(sorted_rows, start=1):
        row["queue_position"] = idx
    return sorted_rows


def render_simulation_tab() -> None:
    _section_header("Real-time Triage Simulation", "Simulate random arrivals, queue reordering, and department load over time.")

    dataset_path = ROOT / "data" / "synthetic_patients.csv"
    if not dataset_path.exists():
        st.warning("Dataset not found. Generate or restore data/synthetic_patients.csv first.")
        return

    pool_df = pd.read_csv(dataset_path)
    if pool_df.empty:
        st.warning("Simulation pool is empty.")
        return

    if "sim_state" not in st.session_state:
        st.session_state["sim_state"] = _sim_default_state()

    sim_state: Dict[str, object] = st.session_state["sim_state"]

    controls = st.container(border=True)
    c1, c2, c3 = controls.columns(3)
    with c1:
        steps = st.slider("Minutes to advance", min_value=1, max_value=120, value=15)
    with c2:
        arrivals_lambda = st.slider("Avg arrivals/min (Poisson λ)", min_value=0.2, max_value=4.0, value=1.5, step=0.1)
    with c3:
        seed = st.number_input("Simulation seed", min_value=1, max_value=999999, value=42, step=1)

    run_col, reset_col = st.columns(2)
    run_sim = run_col.button("Run Simulation Step", type="primary", width="stretch")
    reset_sim = reset_col.button("Reset Simulation", width="stretch")

    if reset_sim:
        st.session_state["sim_state"] = _sim_default_state()
        st.rerun()

    if run_sim:
        rng = np.random.default_rng(seed + int(sim_state["current_minute"]))
        engine = get_engine()

        department_capacity = {
            "Emergency Department": 2,
            "Cardiology": 1,
            "Neurology": 1,
            "Pulmonology": 1,
            "Orthopedics": 1,
            "Gastroenterology": 1,
            "Pediatrics": 1,
            "General Medicine": 2,
        }

        queue_rows = list(sim_state["queue"])
        timeline_rows = list(sim_state["timeline"])
        current_minute = int(sim_state["current_minute"])
        arrived = int(sim_state["arrived"])
        completed = int(sim_state["completed"])

        for _ in range(steps):
            current_minute += 1

            arrivals_count = int(rng.poisson(arrivals_lambda))
            if arrivals_count > 0:
                sampled = pool_df.sample(n=arrivals_count, replace=True, random_state=int(rng.integers(0, 10_000_000)))
                for _, row in sampled.iterrows():
                    patient_payload = row.to_dict()
                    patient_payload["Patient_ID"] = f"SIM-{current_minute}-{arrived + 1}"

                    pred = engine.predict_one(patient_payload)
                    queue_rows.append(
                        {
                            "patient_id": pred["patient_id"],
                            "age": int(patient_payload["Age"]),
                            "risk_level": pred["risk_level"],
                            "priority_score": float(pred["priority_score"]),
                            "department": pred["department"],
                            "arrival_minute": current_minute,
                            "queue_position": 0,
                        }
                    )
                    arrived += 1

            queue_rows = _sort_sim_queue(queue_rows)

            for department, capacity in department_capacity.items():
                served = 0
                retained_rows: List[Dict[str, object]] = []
                for item in queue_rows:
                    if item["department"] == department and served < capacity:
                        served += 1
                        completed += 1
                        continue
                    retained_rows.append(item)
                queue_rows = retained_rows

            queue_rows = _sort_sim_queue(queue_rows)

            if queue_rows:
                load_snapshot = pd.DataFrame(queue_rows)["department"].value_counts().to_dict()
            else:
                load_snapshot = {}

            for department in department_capacity:
                timeline_rows.append(
                    {
                        "minute": current_minute,
                        "department": department,
                        "waiting": int(load_snapshot.get(department, 0)),
                    }
                )

        sim_state["current_minute"] = current_minute
        sim_state["queue"] = queue_rows
        sim_state["timeline"] = timeline_rows
        sim_state["arrived"] = arrived
        sim_state["completed"] = completed
        st.session_state["sim_state"] = sim_state

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Simulated Minute", int(sim_state["current_minute"]))
    m2.metric("Patients Arrived", int(sim_state["arrived"]))
    m3.metric("Patients Completed", int(sim_state["completed"]))
    m4.metric("Currently Waiting", len(sim_state["queue"]))

    queue_df = pd.DataFrame(sim_state["queue"])
    if not queue_df.empty:
        st.markdown("#### Dynamic Queue (Reordered by Priority)")
        st.dataframe(
            queue_df[["queue_position", "patient_id", "risk_level", "priority_score", "department", "arrival_minute"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Queue is currently empty. Run simulation steps to generate arrivals.")

    timeline_df = pd.DataFrame(sim_state["timeline"])
    if not timeline_df.empty:
        fig_load = px.line(
            timeline_df,
            x="minute",
            y="waiting",
            color="department",
            title="Department Load Over Time",
            labels={"minute": "Simulation Minute", "waiting": "Waiting Patients", "department": "Department"},
        )
        st.plotly_chart(_style_plot(fig_load), width="stretch")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1280px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.01em;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--secondary-background-color);
            border-radius: 0.75rem;
            padding: 0.4rem 0.75rem;
            background: var(--background-color);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 0.6rem;
            min-height: 2.5rem;
            font-weight: 600;
        }

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"],
        .stMultiSelect [data-baseweb="select"],
        .stDateInput input,
        .stTextArea textarea {
            border-radius: 0.6rem !important;
        }

        button[data-baseweb="tab"] {
            border-radius: 0.6rem 0.6rem 0 0;
            font-weight: 600;
            padding: 0.5rem 0.8rem;
        }

        [data-testid="stDataFrame"] {
            border-radius: 0.6rem;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="AI Patient Triage System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    st.title("🏥 AI Patient Triage System")
    st.caption("AI-powered triage workspace.")

    db = get_db()
    queue_snapshot = pd.DataFrame(db.get_priority_queue())
    prediction_snapshot = pd.DataFrame(db.get_predictions(limit=5000))

    waiting_snapshot = (
        queue_snapshot[queue_snapshot["status"] == "waiting"]
        if not queue_snapshot.empty and "status" in queue_snapshot.columns
        else pd.DataFrame()
    )
    high_snapshot = (
        int((prediction_snapshot["risk_level"] == "High").sum())
        if not prediction_snapshot.empty and "risk_level" in prediction_snapshot.columns
        else 0
    )
    departments_snapshot = (
        int(prediction_snapshot["recommended_department"].nunique())
        if not prediction_snapshot.empty and "recommended_department" in prediction_snapshot.columns
        else 0
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Predictions", int(len(prediction_snapshot)))
    s2.metric("Queue Waiting", int(len(waiting_snapshot)))
    s3.metric("High-Risk Cases", high_snapshot)
    s4.metric("Active Departments", departments_snapshot)

    with st.sidebar:
        st.markdown("### System Status")
        if st.button("Run Health Check", width="stretch"):
            status = run_healthcheck(ROOT)
            if status.ok:
                st.success("All health checks passed")
            else:
                st.error("One or more health checks failed")
            st.dataframe(healthcheck_dataframe(ROOT), width="stretch", hide_index=True)

        st.markdown("### Quick Guide")
        st.caption("1) Analyze\n2) Manage queue\n3) Review analytics")
        st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "🧑‍⚕️ Single Analysis",
            "📥 Batch Processing",
            "📋 Queue Management",
            "📊 Analytics",
            "🗂️ Patient History",
            "📈 Model Performance",
            "⚖️ Bias & Fairness",
            "⏱️ Triage Simulation",
        ]
    )

    with tab1:
        render_single_patient_tab()
    with tab2:
        render_batch_tab()
    with tab3:
        render_queue_tab()
    with tab4:
        render_analytics_tab()
    with tab5:
        render_history_tab()
    with tab6:
        render_model_performance_tab()
    with tab7:
        render_fairness_tab()
    with tab8:
        render_simulation_tab()


if __name__ == "__main__":
    main()
