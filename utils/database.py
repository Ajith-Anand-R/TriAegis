from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List


class TriageDatabase:
    def __init__(self, db_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.db_path = Path(db_path) if db_path else root / "database" / "patients.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    age INTEGER,
                    gender TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symptoms TEXT,
                    bp_systolic INTEGER,
                    bp_diastolic INTEGER,
                    heart_rate INTEGER,
                    temperature REAL,
                    pre_existing_conditions TEXT,
                    risk_level TEXT,
                    risk_probability_low REAL,
                    risk_probability_medium REAL,
                    risk_probability_high REAL,
                    recommended_department TEXT,
                    department_reason TEXT,
                    priority_score REAL,
                    priority_category TEXT,
                    estimated_wait_time TEXT,
                    model_confidence REAL,
                    processing_time_ms INTEGER,
                    source TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                );

                CREATE TABLE IF NOT EXISTS shap_explanations (
                    explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    feature_name TEXT,
                    feature_value TEXT,
                    shap_value REAL,
                    rank INTEGER,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS priority_queue (
                    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    arrival_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    priority_score REAL,
                    department TEXT,
                    status TEXT DEFAULT 'waiting',
                    queue_position INTEGER,
                    estimated_wait_time TEXT,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                );
                """
            )

    def _upsert_patient(self, conn: sqlite3.Connection, patient: Dict[str, object]) -> None:
        conn.execute(
            """
            INSERT INTO patients (patient_id, age, gender, notes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                age = excluded.age,
                gender = excluded.gender,
                notes = COALESCE(excluded.notes, patients.notes),
                updated_at = CURRENT_TIMESTAMP;
            """,
            (
                patient["Patient_ID"],
                int(patient.get("Age", 0)),
                str(patient.get("Gender", "Unknown")),
                str(patient.get("notes", "")) or None,
            ),
        )

    @staticmethod
    def _bp_parts(blood_pressure: str) -> tuple[int, int]:
        parts = str(blood_pressure).split("/")
        if len(parts) != 2:
            return 120, 80
        return int(parts[0]), int(parts[1])

    def save_prediction(
        self,
        patient: Dict[str, object],
        prediction: Dict[str, object],
        shap_top_contributors: Iterable[Dict[str, object]] | None = None,
        source: str = "manual",
    ) -> int:
        with self.connection() as conn:
            self._upsert_patient(conn, patient)

            bp_s, bp_d = self._bp_parts(str(patient.get("Blood Pressure", "120/80")))
            probs = prediction.get("probabilities", {})

            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    patient_id, symptoms, bp_systolic, bp_diastolic, heart_rate, temperature,
                    pre_existing_conditions, risk_level, risk_probability_low,
                    risk_probability_medium, risk_probability_high, recommended_department,
                    department_reason, priority_score, priority_category, estimated_wait_time,
                    model_confidence, processing_time_ms, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient["Patient_ID"],
                    str(patient.get("Symptoms", "")),
                    bp_s,
                    bp_d,
                    int(patient.get("Heart Rate", 70)),
                    float(patient.get("Temperature", 98.6)),
                    str(patient.get("Pre-Existing Conditions", "none")),
                    str(prediction.get("risk_level", "Low")),
                    float(probs.get("Low", 0.0)),
                    float(probs.get("Medium", 0.0)),
                    float(probs.get("High", 0.0)),
                    str(prediction.get("department", "General Medicine")),
                    str(prediction.get("department_reason", "")),
                    float(prediction.get("priority_score", 1.0)),
                    str(prediction.get("priority_category", "Low")),
                    str(prediction.get("estimated_wait_time", "60 minutes")),
                    float(prediction.get("confidence", 0.0)),
                    int(prediction.get("processing_time_ms", 0)),
                    source,
                ),
            )
            prediction_id = int(cursor.lastrowid)

            if shap_top_contributors:
                rows = [
                    (
                        prediction_id,
                        str(item.get("feature", "")),
                        str(item.get("value", "")),
                        float(item.get("impact", 0.0)),
                        int(item.get("rank", index + 1)),
                    )
                    for index, item in enumerate(shap_top_contributors)
                ]
                conn.executemany(
                    """
                    INSERT INTO shap_explanations (
                        prediction_id, feature_name, feature_value, shap_value, rank
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )

            conn.execute(
                """
                INSERT INTO priority_queue (
                    prediction_id, priority_score, department, status, queue_position, estimated_wait_time
                ) VALUES (?, ?, ?, 'waiting', ?, ?)
                """,
                (
                    prediction_id,
                    float(prediction.get("priority_score", 1.0)),
                    str(prediction.get("department", "General Medicine")),
                    int(prediction.get("queue_position", 1)),
                    str(prediction.get("estimated_wait_time", "60 minutes")),
                ),
            )

            self.reorder_queue(conn)
            return prediction_id

    def reorder_queue(self, conn: sqlite3.Connection | None = None) -> None:
        owns_connection = conn is None
        if owns_connection:
            context = self.connection()
            conn = context.__enter__()

        try:
            rows = conn.execute(
                """
                SELECT q.queue_id, q.priority_score, q.arrival_time, p.age
                FROM priority_queue q
                JOIN predictions pr ON pr.prediction_id = q.prediction_id
                JOIN patients p ON p.patient_id = pr.patient_id
                WHERE q.status = 'waiting'
                ORDER BY q.priority_score DESC, p.age DESC, q.arrival_time ASC
                """
            ).fetchall()

            for position, row in enumerate(rows, start=1):
                conn.execute(
                    "UPDATE priority_queue SET queue_position = ? WHERE queue_id = ?",
                    (position, row["queue_id"]),
                )
        finally:
            if owns_connection:
                context.__exit__(None, None, None)

    def get_priority_queue(self, status: str | None = None) -> List[Dict[str, object]]:
        with self.connection() as conn:
            base_query = """
                SELECT q.queue_id, q.prediction_id, q.arrival_time, q.priority_score,
                       q.department, q.status, q.queue_position, q.estimated_wait_time,
                       pr.risk_level, pr.patient_id, pr.timestamp,
                       pr.bp_systolic, pr.bp_diastolic, pr.heart_rate, pr.temperature,
                       pr.model_confidence
                FROM priority_queue q
                JOIN predictions pr ON pr.prediction_id = q.prediction_id
            """
            params: tuple = ()
            if status:
                base_query += " WHERE q.status = ?"
                params = (status,)
            base_query += " ORDER BY q.queue_position ASC"

            rows = conn.execute(base_query, params).fetchall()
            return [dict(row) for row in rows]

    def apply_continuous_monitoring(self, check_interval_minutes: int = 5) -> List[Dict[str, object]]:
        alerts: List[Dict[str, object]] = []

        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT q.queue_id, q.priority_score, q.arrival_time,
                       pr.patient_id, pr.bp_systolic, pr.bp_diastolic,
                       pr.heart_rate, pr.temperature
                FROM priority_queue q
                JOIN predictions pr ON pr.prediction_id = q.prediction_id
                WHERE q.status = 'waiting'
                """
            ).fetchall()

            for row in rows:
                arrival_raw = str(row["arrival_time"]) if row["arrival_time"] is not None else ""
                try:
                    arrival = datetime.fromisoformat(arrival_raw.replace(" ", "T")).replace(tzinfo=UTC)
                except ValueError:
                    continue

                elapsed_minutes = max(0, int((datetime.now(UTC) - arrival).total_seconds() // 60))

                bp_s = int(row["bp_systolic"] or 120)
                bp_d = int(row["bp_diastolic"] or 80)
                heart_rate = int(row["heart_rate"] or 70)
                temperature = float(row["temperature"] or 98.6)

                worsening_signals = 0
                if bp_s < 90 or bp_s > 180:
                    worsening_signals += 1
                if bp_d > 110:
                    worsening_signals += 1
                if heart_rate < 45 or heart_rate > 130:
                    worsening_signals += 1
                if temperature >= 102.5:
                    worsening_signals += 1

                review_cycles = elapsed_minutes // max(check_interval_minutes, 1)
                escalation = min((review_cycles * 0.1) + (max(0, worsening_signals - 1) * 0.2), 1.5)

                current_priority = float(row["priority_score"] or 1.0)
                updated_priority = round(min(10.0, current_priority + escalation), 1)

                if updated_priority >= current_priority + 0.5:
                    conn.execute(
                        "UPDATE priority_queue SET priority_score = ? WHERE queue_id = ?",
                        (updated_priority, int(row["queue_id"])),
                    )
                    alerts.append(
                        {
                            "patient_id": str(row["patient_id"]),
                            "message": (
                                f"Patient {row['patient_id']} deterioration detected: "
                                f"priority escalated {current_priority:.1f} → {updated_priority:.1f}"
                            ),
                            "severity": "high" if updated_priority >= 9.0 else "medium",
                        }
                    )

            self.reorder_queue(conn)

        return alerts

    def update_queue_status(self, queue_id: int, status: str) -> None:
        if status not in {"waiting", "in_progress", "completed"}:
            raise ValueError("Invalid queue status")

        with self.connection() as conn:
            conn.execute(
                "UPDATE priority_queue SET status = ? WHERE queue_id = ?",
                (status, queue_id),
            )
            self.reorder_queue(conn)

    def clear_completed(self) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM priority_queue
                WHERE status = 'completed'
                  AND arrival_time < datetime('now', '-24 hours')
                """
            )
            return int(cursor.rowcount)

    def clear_old_records(self, days: int = 90) -> Dict[str, int]:
        with self.connection() as conn:
            queue_cursor = conn.execute(
                """
                DELETE FROM priority_queue
                WHERE arrival_time < datetime('now', ?)
                """,
                (f"-{int(days)} days",),
            )
            prediction_cursor = conn.execute(
                """
                DELETE FROM predictions
                WHERE timestamp < datetime('now', ?)
                """,
                (f"-{int(days)} days",),
            )
            return {
                "deleted_queue_rows": int(queue_cursor.rowcount),
                "deleted_prediction_rows": int(prediction_cursor.rowcount),
            }

    def get_high_risk_patients(self, limit: int = 100) -> List[Dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                WHERE risk_level = 'High'
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_department_load(self) -> List[Dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT department,
                       COUNT(*) AS waiting_count,
                       AVG(priority_score) AS avg_priority
                FROM priority_queue
                WHERE status = 'waiting'
                GROUP BY department
                ORDER BY waiting_count DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_patient_history(self, patient_id: str) -> List[Dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM predictions
                WHERE patient_id = ?
                ORDER BY timestamp DESC
                """,
                (patient_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_predictions(self, limit: int = 1000) -> List[Dict[str, object]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT prediction_id, patient_id, timestamp, symptoms, bp_systolic, bp_diastolic,
                       heart_rate, temperature, pre_existing_conditions, risk_level,
                       risk_probability_low, risk_probability_medium, risk_probability_high,
                       recommended_department, department_reason, priority_score,
                       priority_category, estimated_wait_time, model_confidence,
                       processing_time_ms, source
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def search_predictions(
        self,
        patient_id_query: str | None = None,
        risk_levels: List[str] | None = None,
        departments: List[str] | None = None,
        priority_categories: List[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 2000,
    ) -> List[Dict[str, object]]:
        query = [
            """
            SELECT prediction_id, patient_id, timestamp, symptoms, bp_systolic, bp_diastolic,
                   heart_rate, temperature, pre_existing_conditions, risk_level,
                   recommended_department, priority_score, priority_category,
                   estimated_wait_time, model_confidence, source
            FROM predictions
            WHERE 1=1
            """
        ]
        params: List[object] = []

        if patient_id_query:
            query.append("AND patient_id LIKE ?")
            params.append(f"%{patient_id_query}%")

        if risk_levels:
            placeholders = ",".join(["?"] * len(risk_levels))
            query.append(f"AND risk_level IN ({placeholders})")
            params.extend(risk_levels)

        if departments:
            placeholders = ",".join(["?"] * len(departments))
            query.append(f"AND recommended_department IN ({placeholders})")
            params.extend(departments)

        if priority_categories:
            placeholders = ",".join(["?"] * len(priority_categories))
            query.append(f"AND priority_category IN ({placeholders})")
            params.extend(priority_categories)

        if start_date:
            query.append("AND date(timestamp) >= date(?)")
            params.append(start_date)

        if end_date:
            query.append("AND date(timestamp) <= date(?)")
            params.append(end_date)

        query.append("ORDER BY timestamp DESC LIMIT ?")
        params.append(limit)

        with self.connection() as conn:
            rows = conn.execute("\n".join(query), tuple(params)).fetchall()
            return [dict(row) for row in rows]

    def delete_specific_records(
        self,
        patient_id: str | None = None,
        prediction_id: int | None = None,
        queue_id: int | None = None,
    ) -> Dict[str, int]:
        deleted_queue_rows = 0
        deleted_prediction_rows = 0
        deleted_patient_rows = 0

        with self.connection() as conn:
            if queue_id is not None:
                cursor = conn.execute(
                    "DELETE FROM priority_queue WHERE queue_id = ?",
                    (int(queue_id),),
                )
                deleted_queue_rows += int(cursor.rowcount)

            if prediction_id is not None:
                queue_cursor = conn.execute(
                    "DELETE FROM priority_queue WHERE prediction_id = ?",
                    (int(prediction_id),),
                )
                deleted_queue_rows += int(queue_cursor.rowcount)

                prediction_cursor = conn.execute(
                    "DELETE FROM predictions WHERE prediction_id = ?",
                    (int(prediction_id),),
                )
                deleted_prediction_rows += int(prediction_cursor.rowcount)

            if patient_id is not None:
                queue_cursor = conn.execute(
                    """
                    DELETE FROM priority_queue
                    WHERE prediction_id IN (
                        SELECT prediction_id FROM predictions WHERE patient_id = ?
                    )
                    """,
                    (str(patient_id),),
                )
                deleted_queue_rows += int(queue_cursor.rowcount)

                prediction_cursor = conn.execute(
                    "DELETE FROM predictions WHERE patient_id = ?",
                    (str(patient_id),),
                )
                deleted_prediction_rows += int(prediction_cursor.rowcount)

                patient_cursor = conn.execute(
                    "DELETE FROM patients WHERE patient_id = ?",
                    (str(patient_id),),
                )
                deleted_patient_rows += int(patient_cursor.rowcount)

        return {
            "deleted_queue_rows": deleted_queue_rows,
            "deleted_prediction_rows": deleted_prediction_rows,
            "deleted_patient_rows": deleted_patient_rows,
        }

    def delete_recent_records(self, days: int = 30, scope: str = "all") -> Dict[str, int]:
        if days <= 0:
            raise ValueError("days must be greater than 0")

        valid_scopes = {"all", "queue", "predictions", "patients"}
        if scope not in valid_scopes:
            raise ValueError(f"Invalid scope. Use one of: {', '.join(sorted(valid_scopes))}")

        deleted_queue_rows = 0
        deleted_prediction_rows = 0
        deleted_patient_rows = 0

        with self.connection() as conn:
            date_clause = f"-{int(days)} days"

            if scope in {"all", "queue"}:
                queue_cursor = conn.execute(
                    """
                    DELETE FROM priority_queue
                    WHERE arrival_time >= datetime('now', ?)
                    """,
                    (date_clause,),
                )
                deleted_queue_rows += int(queue_cursor.rowcount)

            if scope in {"all", "predictions"}:
                queue_for_predictions_cursor = conn.execute(
                    """
                    DELETE FROM priority_queue
                    WHERE prediction_id IN (
                        SELECT prediction_id
                        FROM predictions
                        WHERE timestamp >= datetime('now', ?)
                    )
                    """,
                    (date_clause,),
                )
                deleted_queue_rows += int(queue_for_predictions_cursor.rowcount)

                prediction_cursor = conn.execute(
                    """
                    DELETE FROM predictions
                    WHERE timestamp >= datetime('now', ?)
                    """,
                    (date_clause,),
                )
                deleted_prediction_rows += int(prediction_cursor.rowcount)

            if scope in {"all", "patients"}:
                patient_cursor = conn.execute(
                    """
                    DELETE FROM patients
                    WHERE updated_at >= datetime('now', ?)
                      AND patient_id NOT IN (SELECT DISTINCT patient_id FROM predictions)
                    """,
                    (date_clause,),
                )
                deleted_patient_rows += int(patient_cursor.rowcount)

        return {
            "deleted_queue_rows": deleted_queue_rows,
            "deleted_prediction_rows": deleted_prediction_rows,
            "deleted_patient_rows": deleted_patient_rows,
        }
