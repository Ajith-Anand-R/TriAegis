from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_WARNING_THRESHOLD = 0.75
DEFAULT_CRITICAL_THRESHOLD = 0.90

BED_STATUS_AVAILABLE = "available"
BED_STATUS_OCCUPIED = "occupied"

DEFAULT_OCCUPANCY_SEED: Dict[str, object] = {
    "hospitals": [
        {
            "hospital_id": "HOSP-CENTRAL",
            "name": "Central City Hospital",
            "region": "Central",
            "wards": [
                {
                    "ward_id": "CC-ED",
                    "name": "Emergency Unit",
                    "specialty": "Emergency Department",
                    "capacity": 20,
                    "occupied": 16,
                },
                {
                    "ward_id": "CC-CARD",
                    "name": "Cardiology Ward",
                    "specialty": "Cardiology",
                    "capacity": 14,
                    "occupied": 13,
                },
                {
                    "ward_id": "CC-GEN",
                    "name": "General Medicine Ward",
                    "specialty": "General Medicine",
                    "capacity": 30,
                    "occupied": 17,
                },
            ],
        },
        {
            "hospital_id": "HOSP-NORTH",
            "name": "North Valley Medical Center",
            "region": "North",
            "wards": [
                {
                    "ward_id": "NV-ED",
                    "name": "Emergency Stabilization",
                    "specialty": "Emergency Department",
                    "capacity": 18,
                    "occupied": 17,
                },
                {
                    "ward_id": "NV-NEURO",
                    "name": "Neurology Ward",
                    "specialty": "Neurology",
                    "capacity": 12,
                    "occupied": 9,
                },
                {
                    "ward_id": "NV-GEN",
                    "name": "General Medicine Ward",
                    "specialty": "General Medicine",
                    "capacity": 24,
                    "occupied": 20,
                },
            ],
        },
        {
            "hospital_id": "HOSP-WEST",
            "name": "Westside Care Hospital",
            "region": "West",
            "wards": [
                {
                    "ward_id": "WS-ORTHO",
                    "name": "Orthopedics Ward",
                    "specialty": "Orthopedics",
                    "capacity": 16,
                    "occupied": 9,
                },
                {
                    "ward_id": "WS-PULM",
                    "name": "Pulmonology Ward",
                    "specialty": "Pulmonology",
                    "capacity": 10,
                    "occupied": 9,
                },
                {
                    "ward_id": "WS-GEN",
                    "name": "General Medicine Ward",
                    "specialty": "General Medicine",
                    "capacity": 22,
                    "occupied": 14,
                },
            ],
        },
    ]
}


def _deep_copy_seed(payload: Dict[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(payload))


def _normalize_thresholds(warning_threshold: float, critical_threshold: float) -> tuple[float, float]:
    warning = float(warning_threshold)
    critical = float(critical_threshold)

    if warning <= 0 or warning >= 1:
        warning = DEFAULT_WARNING_THRESHOLD
    if critical <= 0 or critical > 1:
        critical = DEFAULT_CRITICAL_THRESHOLD
    if critical <= warning:
        warning = DEFAULT_WARNING_THRESHOLD
        critical = DEFAULT_CRITICAL_THRESHOLD

    return warning, critical


def _parse_timestamp_utc(timestamp_value: str | None) -> datetime:
    if not timestamp_value:
        return datetime.now(UTC)

    raw = str(timestamp_value).strip()
    if not raw:
        return datetime.now(UTC)

    normalized = raw.replace("Z", "+00:00").replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.now(UTC)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def calculate_occupancy_status(
    occupied_beds: int,
    capacity: int,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> str:
    if capacity <= 0:
        raise ValueError("capacity must be greater than 0")

    warning, critical = _normalize_thresholds(warning_threshold, critical_threshold)
    ratio = float(occupied_beds) / float(capacity)

    if ratio >= critical:
        return "Critical"
    if ratio >= warning:
        return "Warning"
    return "OK"


def load_occupancy_seed(seed_path: str | Path | None = None) -> Dict[str, object]:
    if seed_path is None:
        return _deep_copy_seed(DEFAULT_OCCUPANCY_SEED)

    path = Path(seed_path)
    if not path.exists():
        return _deep_copy_seed(DEFAULT_OCCUPANCY_SEED)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("hospitals"), list):
        raise ValueError("Occupancy seed file must contain a top-level 'hospitals' list")
    return payload


class OccupancyEngine:
    def __init__(self, db_path: str | Path | None = None, seed_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.db_path = Path(db_path) if db_path else root / "database" / "patients.db"
        self.seed_path = Path(seed_path) if seed_path else root / "data" / "occupancy_seed.json"
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
                CREATE TABLE IF NOT EXISTS hospitals (
                    hospital_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    region TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS wards (
                    ward_id TEXT PRIMARY KEY,
                    hospital_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    specialty TEXT NOT NULL,
                    capacity INTEGER NOT NULL CHECK(capacity >= 1),
                    warning_threshold REAL NOT NULL DEFAULT 0.75,
                    critical_threshold REAL NOT NULL DEFAULT 0.90,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS beds (
                    bed_id TEXT PRIMARY KEY,
                    ward_id TEXT NOT NULL,
                    bed_label TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'available',
                    current_patient_id TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ward_id) REFERENCES wards(ward_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_beds_ward_status ON beds(ward_id, status);

                CREATE TABLE IF NOT EXISTS occupancy_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    hospital_id TEXT,
                    ward_id TEXT,
                    bed_id TEXT,
                    patient_id TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id) ON DELETE SET NULL,
                    FOREIGN KEY (ward_id) REFERENCES wards(ward_id) ON DELETE SET NULL,
                    FOREIGN KEY (bed_id) REFERENCES beds(bed_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS routing_decisions (
                    route_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    priority_score REAL NOT NULL,
                    department TEXT NOT NULL,
                    preferred_hospital_id TEXT,
                    requested_queue_ahead INTEGER NOT NULL DEFAULT 0,
                    recommended_hospital_id TEXT NOT NULL,
                    recommended_ward_id TEXT NOT NULL,
                    estimated_wait_minutes INTEGER NOT NULL DEFAULT 0,
                    has_capacity INTEGER NOT NULL DEFAULT 0,
                    overflow_risk TEXT NOT NULL DEFAULT 'none',
                    accepted INTEGER NOT NULL DEFAULT 0,
                    admitted INTEGER NOT NULL DEFAULT 0,
                    admitted_bed_id TEXT,
                    admitted_at TIMESTAMP,
                    actual_wait_minutes REAL,
                    wait_delta_minutes REAL,
                    discharged INTEGER NOT NULL DEFAULT 0,
                    discharged_at TIMESTAMP,
                    route_reason TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_routing_decisions_patient
                ON routing_decisions(patient_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_routing_decisions_created
                ON routing_decisions(created_at DESC);
                """
            )

    def seed_from_file(self, reset: bool = False) -> Dict[str, int]:
        payload = load_occupancy_seed(self.seed_path)
        return self.seed_from_payload(payload, reset=reset)

    def seed_from_payload(self, payload: Dict[str, object], reset: bool = False) -> Dict[str, int]:
        hospitals = payload.get("hospitals", [])
        if not isinstance(hospitals, list):
            raise ValueError("Seed payload must include a list of hospitals")

        hospital_count = 0
        ward_count = 0
        bed_count = 0

        with self.connection() as conn:
            if reset:
                conn.execute("DELETE FROM occupancy_events")
                conn.execute("DELETE FROM routing_decisions")
                conn.execute("DELETE FROM beds")
                conn.execute("DELETE FROM wards")
                conn.execute("DELETE FROM hospitals")

            for hospital in hospitals:
                if not isinstance(hospital, dict):
                    raise ValueError("Each hospital entry must be an object")

                hospital_id = str(hospital.get("hospital_id", "")).strip()
                if not hospital_id:
                    raise ValueError("hospital_id is required for every hospital")

                hospital_name = str(hospital.get("name", hospital_id)).strip() or hospital_id
                region = str(hospital.get("region", "")).strip() or None

                conn.execute(
                    """
                    INSERT INTO hospitals (hospital_id, name, region)
                    VALUES (?, ?, ?)
                    ON CONFLICT(hospital_id) DO UPDATE SET
                        name = excluded.name,
                        region = excluded.region
                    """,
                    (hospital_id, hospital_name, region),
                )
                hospital_count += 1

                wards = hospital.get("wards", [])
                if not isinstance(wards, list):
                    raise ValueError(f"Hospital {hospital_id} has invalid wards list")

                for ward in wards:
                    if not isinstance(ward, dict):
                        raise ValueError(f"Hospital {hospital_id} includes an invalid ward entry")

                    ward_id = str(ward.get("ward_id", "")).strip()
                    if not ward_id:
                        raise ValueError(f"Hospital {hospital_id} has ward without ward_id")

                    ward_name = str(ward.get("name", ward_id)).strip() or ward_id
                    specialty = str(ward.get("specialty", "General Medicine")).strip() or "General Medicine"
                    capacity = max(1, int(ward.get("capacity", 1)))
                    occupied = max(0, min(capacity, int(ward.get("occupied", 0))))

                    warning, critical = _normalize_thresholds(
                        float(ward.get("warning_threshold", DEFAULT_WARNING_THRESHOLD)),
                        float(ward.get("critical_threshold", DEFAULT_CRITICAL_THRESHOLD)),
                    )

                    conn.execute(
                        """
                        INSERT INTO wards (
                            ward_id, hospital_id, name, specialty, capacity,
                            warning_threshold, critical_threshold
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ward_id) DO UPDATE SET
                            hospital_id = excluded.hospital_id,
                            name = excluded.name,
                            specialty = excluded.specialty,
                            capacity = excluded.capacity,
                            warning_threshold = excluded.warning_threshold,
                            critical_threshold = excluded.critical_threshold
                        """,
                        (ward_id, hospital_id, ward_name, specialty, capacity, warning, critical),
                    )

                    conn.execute("DELETE FROM beds WHERE ward_id = ?", (ward_id,))
                    for idx in range(1, capacity + 1):
                        bed_id = f"{ward_id}-B{idx:03d}"
                        bed_label = f"B{idx:03d}"
                        status = BED_STATUS_OCCUPIED if idx <= occupied else BED_STATUS_AVAILABLE
                        current_patient_id = f"SEED-{ward_id}-{idx:03d}" if status == BED_STATUS_OCCUPIED else None

                        conn.execute(
                            """
                            INSERT INTO beds (bed_id, ward_id, bed_label, status, current_patient_id)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (bed_id, ward_id, bed_label, status, current_patient_id),
                        )
                        bed_count += 1

                    ward_count += 1

        return {
            "hospital_count": hospital_count,
            "ward_count": ward_count,
            "bed_count": bed_count,
        }

    def _ward_row_to_snapshot(self, row: sqlite3.Row) -> Dict[str, object]:
        capacity = max(1, int(row["capacity"]))
        occupied_beds = int(row["occupied_beds"] or 0)
        available_beds = int(row["available_beds"] or 0)

        if occupied_beds + available_beds < capacity:
            available_beds = capacity - occupied_beds

        warning = float(row["warning_threshold"])
        critical = float(row["critical_threshold"])
        warning, critical = _normalize_thresholds(warning, critical)

        load_ratio = float(occupied_beds) / float(capacity)
        load_status = calculate_occupancy_status(
            occupied_beds=occupied_beds,
            capacity=capacity,
            warning_threshold=warning,
            critical_threshold=critical,
        )

        return {
            "hospital_id": str(row["hospital_id"]),
            "hospital_name": str(row["hospital_name"]),
            "hospital_region": str(row["hospital_region"] or ""),
            "ward_id": str(row["ward_id"]),
            "ward_name": str(row["ward_name"]),
            "specialty": str(row["specialty"]),
            "capacity": capacity,
            "occupied_beds": occupied_beds,
            "available_beds": available_beds,
            "load_ratio": round(load_ratio, 4),
            "load_percent": round(load_ratio * 100.0, 1),
            "load_status": load_status,
            "warning_threshold": round(warning, 3),
            "critical_threshold": round(critical, 3),
        }

    def _get_ward_snapshot_with_conn(self, conn: sqlite3.Connection, ward_id: str) -> Dict[str, object]:
        row = conn.execute(
            """
            SELECT
                w.ward_id,
                w.hospital_id,
                w.name AS ward_name,
                w.specialty,
                w.capacity,
                w.warning_threshold,
                w.critical_threshold,
                h.name AS hospital_name,
                h.region AS hospital_region,
                COALESCE(SUM(CASE WHEN b.status = 'occupied' THEN 1 ELSE 0 END), 0) AS occupied_beds,
                COALESCE(SUM(CASE WHEN b.status = 'available' THEN 1 ELSE 0 END), 0) AS available_beds
            FROM wards w
            JOIN hospitals h ON h.hospital_id = w.hospital_id
            LEFT JOIN beds b ON b.ward_id = w.ward_id
            WHERE w.ward_id = ?
            GROUP BY
                w.ward_id,
                w.hospital_id,
                w.name,
                w.specialty,
                w.capacity,
                w.warning_threshold,
                w.critical_threshold,
                h.name,
                h.region
            """,
            (ward_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"Ward '{ward_id}' not found")
        return self._ward_row_to_snapshot(row)

    def get_ward_snapshot(self, ward_id: str) -> Dict[str, object]:
        with self.connection() as conn:
            return self._get_ward_snapshot_with_conn(conn, ward_id=ward_id)

    def get_ward_occupancy(self, hospital_id: str | None = None) -> List[Dict[str, object]]:
        with self.connection() as conn:
            query = [
                """
                SELECT
                    w.ward_id,
                    w.hospital_id,
                    w.name AS ward_name,
                    w.specialty,
                    w.capacity,
                    w.warning_threshold,
                    w.critical_threshold,
                    h.name AS hospital_name,
                    h.region AS hospital_region,
                    COALESCE(SUM(CASE WHEN b.status = 'occupied' THEN 1 ELSE 0 END), 0) AS occupied_beds,
                    COALESCE(SUM(CASE WHEN b.status = 'available' THEN 1 ELSE 0 END), 0) AS available_beds
                FROM wards w
                JOIN hospitals h ON h.hospital_id = w.hospital_id
                LEFT JOIN beds b ON b.ward_id = w.ward_id
                WHERE 1=1
                """
            ]
            params: List[object] = []

            if hospital_id:
                query.append("AND w.hospital_id = ?")
                params.append(str(hospital_id))

            query.append(
                """
                GROUP BY
                    w.ward_id,
                    w.hospital_id,
                    w.name,
                    w.specialty,
                    w.capacity,
                    w.warning_threshold,
                    w.critical_threshold,
                    h.name,
                    h.region
                ORDER BY h.name ASC, w.name ASC
                """
            )

            rows = conn.execute("\n".join(query), tuple(params)).fetchall()
            return [self._ward_row_to_snapshot(row) for row in rows]

    def get_hospital_occupancy(self) -> List[Dict[str, object]]:
        wards = self.get_ward_occupancy()
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for ward in wards:
            grouped.setdefault(str(ward["hospital_id"]), []).append(ward)

        with self.connection() as conn:
            hospital_rows = conn.execute(
                """
                SELECT hospital_id, name, region
                FROM hospitals
                ORDER BY name ASC
                """
            ).fetchall()

        hospital_snapshots: List[Dict[str, object]] = []
        for row in hospital_rows:
            hospital_id = str(row["hospital_id"])
            hospital_wards = grouped.get(hospital_id, [])

            total_capacity = int(sum(int(ward["capacity"]) for ward in hospital_wards))
            occupied_beds = int(sum(int(ward["occupied_beds"]) for ward in hospital_wards))
            available_beds = max(0, total_capacity - occupied_beds)

            load_ratio = 0.0 if total_capacity == 0 else float(occupied_beds) / float(total_capacity)
            load_status = "OK"
            if total_capacity > 0:
                load_status = calculate_occupancy_status(occupied_beds, total_capacity)

            hospital_snapshots.append(
                {
                    "hospital_id": hospital_id,
                    "hospital_name": str(row["name"]),
                    "hospital_region": str(row["region"] or ""),
                    "ward_count": len(hospital_wards),
                    "total_capacity": total_capacity,
                    "occupied_beds": occupied_beds,
                    "available_beds": available_beds,
                    "load_ratio": round(load_ratio, 4),
                    "load_percent": round(load_ratio * 100.0, 1),
                    "load_status": load_status,
                    "critical_ward_count": sum(1 for ward in hospital_wards if ward["load_status"] == "Critical"),
                    "warning_ward_count": sum(1 for ward in hospital_wards if ward["load_status"] == "Warning"),
                }
            )

        return hospital_snapshots

    def current_load(self) -> Dict[str, object]:
        wards = self.get_ward_occupancy()
        hospitals = self.get_hospital_occupancy()

        total_capacity = sum(int(item["capacity"]) for item in wards)
        total_occupied = sum(int(item["occupied_beds"]) for item in wards)
        total_available = max(0, total_capacity - total_occupied)
        network_load_ratio = 0.0 if total_capacity == 0 else float(total_occupied) / float(total_capacity)

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": {
                "hospital_count": len(hospitals),
                "ward_count": len(wards),
                "total_capacity": total_capacity,
                "total_occupied": total_occupied,
                "total_available": total_available,
                "network_load_ratio": round(network_load_ratio, 4),
                "network_load_percent": round(network_load_ratio * 100.0, 1),
                "critical_ward_count": sum(1 for ward in wards if ward["load_status"] == "Critical"),
                "warning_ward_count": sum(1 for ward in wards if ward["load_status"] == "Warning"),
            },
            "hospitals": hospitals,
            "wards": wards,
        }

    def reserve_bed(self, ward_id: str, patient_id: str, route_id: str | None = None) -> Dict[str, object]:
        normalized_ward_id = str(ward_id).strip()
        normalized_patient_id = str(patient_id).strip()

        if not normalized_ward_id:
            raise ValueError("ward_id is required")
        if not normalized_patient_id:
            raise ValueError("patient_id is required")

        with self.connection() as conn:
            ward_row = conn.execute(
                """
                SELECT ward_id, hospital_id
                FROM wards
                WHERE ward_id = ?
                """,
                (normalized_ward_id,),
            ).fetchone()
            if ward_row is None:
                raise ValueError(f"Ward '{normalized_ward_id}' not found")

            bed_row = conn.execute(
                """
                SELECT bed_id, bed_label
                FROM beds
                WHERE ward_id = ?
                  AND status = 'available'
                ORDER BY bed_label ASC
                LIMIT 1
                """,
                (normalized_ward_id,),
            ).fetchone()

            if bed_row is None:
                raise ValueError(f"No available bed in ward '{normalized_ward_id}'")

            bed_id = str(bed_row["bed_id"])
            bed_label = str(bed_row["bed_label"])
            hospital_id = str(ward_row["hospital_id"])

            conn.execute(
                """
                UPDATE beds
                SET status = ?, current_patient_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE bed_id = ?
                """,
                (BED_STATUS_OCCUPIED, normalized_patient_id, bed_id),
            )

            conn.execute(
                """
                INSERT INTO occupancy_events (
                    event_type, hospital_id, ward_id, bed_id, patient_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "reserve",
                    hospital_id,
                    normalized_ward_id,
                    bed_id,
                    normalized_patient_id,
                    json.dumps(
                        {
                            "source": "occupancy_engine",
                            "route_id": (str(route_id).strip() if route_id else None),
                        }
                    ),
                ),
            )

            ward_snapshot = self._get_ward_snapshot_with_conn(conn, ward_id=normalized_ward_id)

        return {
            "action": "reserved",
            "hospital_id": hospital_id,
            "ward_id": normalized_ward_id,
            "bed_id": bed_id,
            "bed_label": bed_label,
            "patient_id": normalized_patient_id,
            "ward_snapshot": ward_snapshot,
        }

    def release_bed(
        self,
        bed_id: str,
        patient_id: str | None = None,
        route_id: str | None = None,
    ) -> Dict[str, object]:
        normalized_bed_id = str(bed_id).strip()
        if not normalized_bed_id:
            raise ValueError("bed_id is required")

        expected_patient_id = None if patient_id is None else str(patient_id).strip()

        with self.connection() as conn:
            bed_row = conn.execute(
                """
                SELECT b.bed_id, b.bed_label, b.status, b.current_patient_id, b.ward_id, w.hospital_id
                FROM beds b
                JOIN wards w ON w.ward_id = b.ward_id
                WHERE b.bed_id = ?
                """,
                (normalized_bed_id,),
            ).fetchone()

            if bed_row is None:
                raise ValueError(f"Bed '{normalized_bed_id}' not found")

            status = str(bed_row["status"])
            if status != BED_STATUS_OCCUPIED:
                raise ValueError(f"Bed '{normalized_bed_id}' is not occupied")

            current_patient_id = str(bed_row["current_patient_id"] or "")
            if expected_patient_id and current_patient_id != expected_patient_id:
                raise ValueError(
                    f"Bed '{normalized_bed_id}' is assigned to '{current_patient_id}', not '{expected_patient_id}'"
                )

            ward_id = str(bed_row["ward_id"])
            hospital_id = str(bed_row["hospital_id"])

            conn.execute(
                """
                UPDATE beds
                SET status = ?, current_patient_id = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE bed_id = ?
                """,
                (BED_STATUS_AVAILABLE, normalized_bed_id),
            )

            conn.execute(
                """
                INSERT INTO occupancy_events (
                    event_type, hospital_id, ward_id, bed_id, patient_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "release",
                    hospital_id,
                    ward_id,
                    normalized_bed_id,
                    current_patient_id or expected_patient_id,
                    json.dumps(
                        {
                            "source": "occupancy_engine",
                            "route_id": (str(route_id).strip() if route_id else None),
                        }
                    ),
                ),
            )

            ward_snapshot = self._get_ward_snapshot_with_conn(conn, ward_id=ward_id)

        return {
            "action": "released",
            "hospital_id": hospital_id,
            "ward_id": ward_id,
            "bed_id": normalized_bed_id,
            "bed_label": str(bed_row["bed_label"]),
            "patient_id": current_patient_id,
            "ward_snapshot": ward_snapshot,
        }

    @staticmethod
    def _wait_bucket(wait_minutes: int) -> str:
        if wait_minutes <= 0:
            return "Immediate"
        if wait_minutes <= 15:
            return "Short"
        if wait_minutes <= 45:
            return "Moderate"
        return "Extended"

    def estimate_wait_time(
        self,
        ward_id: str,
        queue_ahead: int = 0,
        average_turnover_minutes: int = 35,
    ) -> Dict[str, object]:
        if average_turnover_minutes <= 0:
            raise ValueError("average_turnover_minutes must be greater than 0")

        queue_ahead = max(0, int(queue_ahead))
        ward_snapshot = self.get_ward_snapshot(ward_id)

        available_beds = int(ward_snapshot["available_beds"])
        load_ratio = float(ward_snapshot["load_ratio"])
        pressure_multiplier = 1.0 + max(0.0, load_ratio - 0.70) * 2.5

        if available_beds > queue_ahead:
            wait_minutes = 0
        else:
            blocked_patients = max(1, queue_ahead - available_beds + 1)
            wait_minutes = int(round(blocked_patients * float(average_turnover_minutes) * pressure_multiplier))

        return {
            "ward_id": str(ward_snapshot["ward_id"]),
            "hospital_id": str(ward_snapshot["hospital_id"]),
            "estimated_wait_minutes": wait_minutes,
            "wait_band": self._wait_bucket(wait_minutes),
            "queue_ahead": queue_ahead,
            "available_beds": available_beds,
            "load_status": str(ward_snapshot["load_status"]),
        }

    def peek_available_bed(self, ward_id: str) -> Dict[str, str] | None:
        normalized_ward_id = str(ward_id).strip()
        if not normalized_ward_id:
            raise ValueError("ward_id is required")

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT bed_id, bed_label
                FROM beds
                WHERE ward_id = ?
                  AND status = 'available'
                ORDER BY bed_label ASC
                LIMIT 1
                """,
                (normalized_ward_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "bed_id": str(row["bed_id"]),
            "bed_label": str(row["bed_label"]),
        }

    def record_route_decision(
        self,
        routing: Dict[str, object],
        queue_ahead: int = 0,
        source: str = "api_route",
        metadata: Dict[str, object] | None = None,
    ) -> str:
        route_id = str(routing.get("route_id") or "").strip() or str(uuid.uuid4())
        normalized_patient_id = str(routing.get("patient_id") or "").strip() or "unknown"

        metadata_payload: Dict[str, Any] = {
            "source": source,
            "alternatives": routing.get("alternatives", []),
            "explanation_fields": routing.get("explanation_fields", {}),
        }
        if metadata:
            metadata_payload.update(metadata)

        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO routing_decisions (
                    route_id,
                    patient_id,
                    risk_level,
                    priority_score,
                    department,
                    preferred_hospital_id,
                    requested_queue_ahead,
                    recommended_hospital_id,
                    recommended_ward_id,
                    estimated_wait_minutes,
                    has_capacity,
                    overflow_risk,
                    route_reason,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(route_id) DO UPDATE SET
                    patient_id = excluded.patient_id,
                    risk_level = excluded.risk_level,
                    priority_score = excluded.priority_score,
                    department = excluded.department,
                    preferred_hospital_id = excluded.preferred_hospital_id,
                    requested_queue_ahead = excluded.requested_queue_ahead,
                    recommended_hospital_id = excluded.recommended_hospital_id,
                    recommended_ward_id = excluded.recommended_ward_id,
                    estimated_wait_minutes = excluded.estimated_wait_minutes,
                    has_capacity = excluded.has_capacity,
                    overflow_risk = excluded.overflow_risk,
                    route_reason = excluded.route_reason,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    route_id,
                    normalized_patient_id,
                    str(routing.get("risk_level") or "Low"),
                    float(routing.get("priority_score") or 1.0),
                    str(routing.get("department") or "General Medicine"),
                    str(routing.get("preferred_hospital_id") or "") or None,
                    max(0, int(queue_ahead)),
                    str(routing.get("recommended_hospital_id") or ""),
                    str(routing.get("recommended_ward_id") or ""),
                    int(routing.get("estimated_wait_minutes") or 0),
                    1 if bool(routing.get("has_capacity")) else 0,
                    str(routing.get("overflow_risk") or "none"),
                    str(routing.get("route_reason") or ""),
                    json.dumps(metadata_payload),
                ),
            )

        return route_id

    def _find_pending_route_row(
        self,
        conn: sqlite3.Connection,
        patient_id: str,
        route_id: str | None = None,
    ) -> sqlite3.Row | None:
        normalized_patient_id = str(patient_id).strip()
        normalized_route_id = str(route_id).strip() if route_id else ""

        if normalized_route_id:
            row = conn.execute(
                """
                SELECT *
                FROM routing_decisions
                WHERE route_id = ? AND patient_id = ?
                """,
                (normalized_route_id, normalized_patient_id),
            ).fetchone()
            if row is not None:
                return row

        return conn.execute(
            """
            SELECT *
            FROM routing_decisions
            WHERE patient_id = ?
              AND admitted = 0
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (normalized_patient_id,),
        ).fetchone()

    def mark_route_admitted(
        self,
        patient_id: str,
        bed_id: str,
        route_id: str | None = None,
    ) -> Dict[str, object] | None:
        normalized_patient_id = str(patient_id).strip()
        normalized_bed_id = str(bed_id).strip()
        if not normalized_patient_id or not normalized_bed_id:
            return None

        with self.connection() as conn:
            row = self._find_pending_route_row(
                conn=conn,
                patient_id=normalized_patient_id,
                route_id=route_id,
            )
            if row is None:
                return None

            created_at = _parse_timestamp_utc(str(row["created_at"] or ""))
            actual_wait_minutes = max(
                0.0, (datetime.now(UTC) - created_at).total_seconds() / 60.0
            )
            estimated_wait_minutes = float(row["estimated_wait_minutes"] or 0.0)
            wait_delta_minutes = actual_wait_minutes - estimated_wait_minutes

            conn.execute(
                """
                UPDATE routing_decisions
                SET
                    accepted = 1,
                    admitted = 1,
                    admitted_bed_id = ?,
                    admitted_at = CURRENT_TIMESTAMP,
                    actual_wait_minutes = ?,
                    wait_delta_minutes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE route_id = ?
                """,
                (
                    normalized_bed_id,
                    round(actual_wait_minutes, 3),
                    round(wait_delta_minutes, 3),
                    str(row["route_id"]),
                ),
            )

            updated = conn.execute(
                """
                SELECT route_id, patient_id, admitted, admitted_bed_id,
                       actual_wait_minutes, wait_delta_minutes, created_at, admitted_at
                FROM routing_decisions
                WHERE route_id = ?
                """,
                (str(row["route_id"]),),
            ).fetchone()

        if updated is None:
            return None

        return {
            "route_id": str(updated["route_id"]),
            "patient_id": str(updated["patient_id"]),
            "admitted": bool(int(updated["admitted"] or 0)),
            "admitted_bed_id": str(updated["admitted_bed_id"] or "") or None,
            "actual_wait_minutes": float(updated["actual_wait_minutes"] or 0.0),
            "wait_delta_minutes": float(updated["wait_delta_minutes"] or 0.0),
            "created_at": str(updated["created_at"]),
            "admitted_at": str(updated["admitted_at"] or ""),
        }

    def mark_route_discharged(
        self,
        patient_id: str,
        bed_id: str | None = None,
        route_id: str | None = None,
    ) -> Dict[str, object] | None:
        normalized_patient_id = str(patient_id).strip()
        if not normalized_patient_id:
            return None

        normalized_route_id = str(route_id).strip() if route_id else ""
        normalized_bed_id = str(bed_id).strip() if bed_id else ""

        with self.connection() as conn:
            if normalized_route_id:
                row = conn.execute(
                    """
                    SELECT *
                    FROM routing_decisions
                    WHERE route_id = ? AND patient_id = ?
                    """,
                    (normalized_route_id, normalized_patient_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT *
                    FROM routing_decisions
                    WHERE patient_id = ?
                      AND admitted = 1
                      AND discharged = 0
                      AND (? = '' OR admitted_bed_id = ?)
                    ORDER BY admitted_at DESC, created_at DESC
                    LIMIT 1
                    """,
                    (normalized_patient_id, normalized_bed_id, normalized_bed_id),
                ).fetchone()

            if row is None:
                return None

            conn.execute(
                """
                UPDATE routing_decisions
                SET
                    discharged = 1,
                    discharged_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE route_id = ?
                """,
                (str(row["route_id"]),),
            )

            updated = conn.execute(
                """
                SELECT route_id, patient_id, discharged, discharged_at, admitted_bed_id
                FROM routing_decisions
                WHERE route_id = ?
                """,
                (str(row["route_id"]),),
            ).fetchone()

        if updated is None:
            return None

        return {
            "route_id": str(updated["route_id"]),
            "patient_id": str(updated["patient_id"]),
            "discharged": bool(int(updated["discharged"] or 0)),
            "discharged_at": str(updated["discharged_at"] or ""),
            "admitted_bed_id": str(updated["admitted_bed_id"] or "") or None,
        }

    def get_routing_operational_metrics(self, hours: int = 24) -> Dict[str, object]:
        window_hours = max(1, int(hours))
        window_clause = f"-{window_hours} hours"

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_routes,
                    SUM(CASE WHEN has_capacity = 1 THEN 1 ELSE 0 END) AS capacity_hits,
                    SUM(CASE WHEN overflow_risk IN ('warning', 'critical') THEN 1 ELSE 0 END) AS overflow_routes,
                    SUM(CASE WHEN admitted = 1 THEN 1 ELSE 0 END) AS admitted_routes,
                    AVG(estimated_wait_minutes) AS mean_estimated_wait,
                    AVG(CASE WHEN admitted = 1 THEN actual_wait_minutes END) AS mean_actual_wait,
                    AVG(CASE WHEN admitted = 1 THEN wait_delta_minutes END) AS mean_wait_delta
                FROM routing_decisions
                WHERE created_at >= datetime('now', ?)
                """,
                (window_clause,),
            ).fetchone()

        total_routes = int((row["total_routes"] if row else 0) or 0)
        capacity_hits = int((row["capacity_hits"] if row else 0) or 0)
        overflow_routes = int((row["overflow_routes"] if row else 0) or 0)
        admitted_routes = int((row["admitted_routes"] if row else 0) or 0)
        mean_estimated_wait = float((row["mean_estimated_wait"] if row else 0.0) or 0.0)
        mean_actual_wait = float((row["mean_actual_wait"] if row else 0.0) or 0.0)
        mean_wait_delta = float((row["mean_wait_delta"] if row else 0.0) or 0.0)

        def pct(value: int, total: int) -> float:
            if total <= 0:
                return 0.0
            return round((float(value) / float(total)) * 100.0, 2)

        return {
            "window_hours": window_hours,
            "total_routes": total_routes,
            "capacity_hit_count": capacity_hits,
            "capacity_hit_rate": pct(capacity_hits, total_routes),
            "overflow_count": overflow_routes,
            "overflow_rate": pct(overflow_routes, total_routes),
            "admitted_count": admitted_routes,
            "admit_conversion_rate": pct(admitted_routes, total_routes),
            "mean_estimated_wait_minutes": round(mean_estimated_wait, 2),
            "mean_actual_wait_minutes": round(mean_actual_wait, 2),
            "mean_wait_delta_minutes": round(mean_wait_delta, 2),
        }

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, object]]:
        limit = max(1, int(limit))
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, hospital_id, ward_id, bed_id, patient_id, metadata_json, created_at
                FROM occupancy_events
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        events: List[Dict[str, object]] = []
        for row in rows:
            metadata_raw = str(row["metadata_json"] or "")
            try:
                metadata = json.loads(metadata_raw) if metadata_raw else {}
            except json.JSONDecodeError:
                metadata = {"raw": metadata_raw}

            events.append(
                {
                    "event_id": int(row["event_id"]),
                    "event_type": str(row["event_type"]),
                    "hospital_id": str(row["hospital_id"] or ""),
                    "ward_id": str(row["ward_id"] or ""),
                    "bed_id": str(row["bed_id"] or ""),
                    "patient_id": str(row["patient_id"] or ""),
                    "metadata": metadata,
                    "created_at": str(row["created_at"]),
                }
            )

        return events
