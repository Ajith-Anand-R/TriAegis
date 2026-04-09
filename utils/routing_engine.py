from __future__ import annotations

import uuid
from collections import Counter
from typing import Dict, List

from utils.occupancy import (
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    OccupancyEngine,
    calculate_occupancy_status,
)


SPECIALTY_FALLBACK_TIERS: Dict[str, List[str]] = {
    "emergency department": ["emergency department", "general medicine"],
    "cardiology": ["cardiology", "emergency department", "general medicine"],
    "neurology": ["neurology", "emergency department", "general medicine"],
    "pulmonology": ["pulmonology", "emergency department", "general medicine"],
    "orthopedics": ["orthopedics", "emergency department", "general medicine"],
    "gastroenterology": ["gastroenterology", "general medicine", "emergency department"],
    "pediatrics": ["pediatrics", "emergency department", "general medicine"],
    "general medicine": ["general medicine", "emergency department"],
}


def _normalize_risk_level(risk_level: str) -> str:
    normalized = str(risk_level).strip().title()
    if normalized not in {"Low", "Medium", "High"}:
        return "Low"
    return normalized


def _risk_rank(risk_level: str) -> int:
    normalized = _normalize_risk_level(risk_level)
    return {"Low": 1, "Medium": 2, "High": 3}[normalized]


def _normalized_key(value: str) -> str:
    return str(value).strip().lower()


def _fallback_tiers(department: str) -> List[str]:
    normalized = _normalized_key(department)
    if normalized in SPECIALTY_FALLBACK_TIERS:
        return SPECIALTY_FALLBACK_TIERS[normalized]
    return [normalized, "emergency department", "general medicine"]


def _specialty_match_score(department: str, ward_specialty: str) -> tuple[float, int | None]:
    dep = _normalized_key(department)
    spec = _normalized_key(ward_specialty)
    tiers = _fallback_tiers(dep)

    if spec in tiers:
        tier_index = tiers.index(spec)
        tier_scores = [1.0, 0.72, 0.55, 0.42]
        if tier_index < len(tier_scores):
            return tier_scores[tier_index], tier_index
        return 0.38, tier_index

    if dep in spec or spec in dep:
        return 0.6, 1

    if spec == "general medicine":
        return 0.5, 2

    if dep in {
        "cardiology",
        "neurology",
        "pulmonology",
        "orthopedics",
        "gastroenterology",
        "pediatrics",
    } and spec == "emergency department":
        return 0.45, 1

    return 0.2, None


def _routing_weights(risk_level: str, priority_score: float) -> Dict[str, float]:
    if risk_level == "High" or priority_score >= 9.0:
        return {
            "availability": 0.42,
            "specialty": 0.30,
            "ward_load": 0.16,
            "hospital_load": 0.12,
        }

    if risk_level == "Medium" or priority_score >= 6.0:
        return {
            "availability": 0.32,
            "specialty": 0.34,
            "ward_load": 0.20,
            "hospital_load": 0.14,
        }

    return {
        "availability": 0.22,
        "specialty": 0.36,
        "ward_load": 0.24,
        "hospital_load": 0.18,
    }


def _transfer_penalty(
    risk_level: str,
    preferred_hospital_id: str | None,
    candidate_hospital_id: str,
) -> float:
    if not preferred_hospital_id or preferred_hospital_id == candidate_hospital_id:
        return 0.0

    if risk_level == "High":
        return 0.06
    if risk_level == "Medium":
        return 0.10
    return 0.14


def _balance_penalty(
    risk_level: str,
    hospital_load_ratio: float,
    min_capacity_hospital_load: float,
    has_other_capacity: bool,
) -> tuple[float, bool]:
    if not has_other_capacity:
        return 0.0, False

    hard_block = (
        risk_level != "High"
        and hospital_load_ratio >= 0.96
        and min_capacity_hospital_load <= 0.86
    )

    if risk_level == "High":
        base_penalty = max(0.0, hospital_load_ratio - 0.93) * 0.22
    elif risk_level == "Medium":
        base_penalty = max(0.0, hospital_load_ratio - 0.86) * 0.34
    else:
        base_penalty = max(0.0, hospital_load_ratio - 0.82) * 0.38

    gap_penalty = max(0.0, (hospital_load_ratio - min_capacity_hospital_load) - 0.04) * 0.15
    blocked_penalty = 0.22 if hard_block else 0.0
    return base_penalty + gap_penalty + blocked_penalty, hard_block


class RoutingEngine:
    def __init__(self, occupancy: OccupancyEngine) -> None:
        self.occupancy = occupancy

    @staticmethod
    def _wait_bucket(wait_minutes: int) -> str:
        if wait_minutes <= 0:
            return "Immediate"
        if wait_minutes <= 15:
            return "Short"
        if wait_minutes <= 45:
            return "Moderate"
        return "Extended"

    @staticmethod
    def _clone_ward_snapshots(wards: List[Dict[str, object]]) -> List[Dict[str, object]]:
        cloned: List[Dict[str, object]] = []
        for ward in wards:
            capacity = max(1, int(ward.get("capacity", 1)))
            occupied = max(0, int(ward.get("occupied_beds", 0)))
            available = max(0, int(ward.get("available_beds", capacity - occupied)))

            if occupied + available > capacity:
                occupied = min(capacity, occupied)
                available = capacity - occupied
            elif occupied + available < capacity:
                available = capacity - occupied

            warning_threshold = float(ward.get("warning_threshold", DEFAULT_WARNING_THRESHOLD))
            critical_threshold = float(ward.get("critical_threshold", DEFAULT_CRITICAL_THRESHOLD))
            load_ratio = float(occupied) / float(capacity)

            cloned.append(
                {
                    "hospital_id": str(ward.get("hospital_id", "")),
                    "hospital_name": str(ward.get("hospital_name", ward.get("hospital_id", ""))),
                    "hospital_region": str(ward.get("hospital_region", "")),
                    "ward_id": str(ward.get("ward_id", "")),
                    "ward_name": str(ward.get("ward_name", ward.get("ward_id", ""))),
                    "specialty": str(ward.get("specialty", "General Medicine")),
                    "capacity": capacity,
                    "occupied_beds": occupied,
                    "available_beds": available,
                    "load_ratio": round(load_ratio, 4),
                    "load_percent": round(load_ratio * 100.0, 1),
                    "load_status": str(
                        ward.get(
                            "load_status",
                            calculate_occupancy_status(
                                occupied,
                                capacity,
                                warning_threshold,
                                critical_threshold,
                            ),
                        )
                    ),
                    "warning_threshold": warning_threshold,
                    "critical_threshold": critical_threshold,
                }
            )

        return cloned

    @staticmethod
    def _build_hospital_snapshots(wards: List[Dict[str, object]]) -> List[Dict[str, object]]:
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for ward in wards:
            grouped.setdefault(str(ward["hospital_id"]), []).append(ward)

        snapshots: List[Dict[str, object]] = []
        for hospital_id, hospital_wards in grouped.items():
            total_capacity = int(sum(int(item["capacity"]) for item in hospital_wards))
            occupied_beds = int(sum(int(item["occupied_beds"]) for item in hospital_wards))
            available_beds = max(0, total_capacity - occupied_beds)

            load_ratio = 0.0 if total_capacity == 0 else float(occupied_beds) / float(total_capacity)
            if total_capacity == 0:
                load_status = "OK"
            else:
                load_status = calculate_occupancy_status(occupied_beds, total_capacity)

            first = hospital_wards[0]
            snapshots.append(
                {
                    "hospital_id": hospital_id,
                    "hospital_name": str(first.get("hospital_name", hospital_id)),
                    "hospital_region": str(first.get("hospital_region", "")),
                    "ward_count": len(hospital_wards),
                    "total_capacity": total_capacity,
                    "occupied_beds": occupied_beds,
                    "available_beds": available_beds,
                    "load_ratio": round(load_ratio, 4),
                    "load_percent": round(load_ratio * 100.0, 1),
                    "load_status": load_status,
                    "critical_ward_count": sum(
                        1 for ward in hospital_wards if str(ward.get("load_status", "")) == "Critical"
                    ),
                    "warning_ward_count": sum(
                        1 for ward in hospital_wards if str(ward.get("load_status", "")) == "Warning"
                    ),
                }
            )

        snapshots.sort(key=lambda item: (str(item["hospital_name"]), str(item["hospital_id"])))
        return snapshots

    @staticmethod
    def _resolve_queue_ahead(payload: Dict[str, object]) -> int:
        raw_queue_ahead = payload.get("queue_ahead")
        if raw_queue_ahead is not None:
            try:
                return max(0, int(raw_queue_ahead))
            except (TypeError, ValueError):
                pass

        raw_queue_position = payload.get("queue_position")
        if raw_queue_position is not None:
            try:
                return max(0, int(raw_queue_position) - 1)
            except (TypeError, ValueError):
                pass

        return 0

    @staticmethod
    def _estimate_wait_for_snapshot(ward: Dict[str, object], queue_ahead: int) -> Dict[str, object]:
        normalized_queue_ahead = max(0, int(queue_ahead))
        available_beds = max(0, int(ward.get("available_beds", 0)))
        load_ratio = float(ward.get("load_ratio", 1.0))
        pressure_multiplier = 1.0 + max(0.0, load_ratio - 0.70) * 2.5

        if available_beds > normalized_queue_ahead:
            wait_minutes = 0
        else:
            blocked_patients = max(1, normalized_queue_ahead - available_beds + 1)
            wait_minutes = int(round(blocked_patients * 35.0 * pressure_multiplier))

        return {
            "estimated_wait_minutes": wait_minutes,
            "wait_band": RoutingEngine._wait_bucket(wait_minutes),
        }

    @staticmethod
    def _consume_virtual_capacity(wards: List[Dict[str, object]], ward_id: str) -> bool:
        normalized_ward_id = str(ward_id).strip()
        for ward in wards:
            if str(ward.get("ward_id", "")) != normalized_ward_id:
                continue

            available = max(0, int(ward.get("available_beds", 0)))
            if available <= 0:
                return False

            capacity = max(1, int(ward.get("capacity", 1)))
            occupied = min(capacity, max(0, int(ward.get("occupied_beds", 0))) + 1)
            updated_available = max(0, capacity - occupied)
            load_ratio = float(occupied) / float(capacity)

            warning_threshold = float(ward.get("warning_threshold", DEFAULT_WARNING_THRESHOLD))
            critical_threshold = float(ward.get("critical_threshold", DEFAULT_CRITICAL_THRESHOLD))

            ward["occupied_beds"] = occupied
            ward["available_beds"] = updated_available
            ward["load_ratio"] = round(load_ratio, 4)
            ward["load_percent"] = round(load_ratio * 100.0, 1)
            ward["load_status"] = calculate_occupancy_status(
                occupied,
                capacity,
                warning_threshold,
                critical_threshold,
            )
            return True

        return False

    def _route_patient_from_snapshots(
        self,
        patient_id: str,
        risk_level: str,
        priority_score: float,
        department: str,
        preferred_hospital_id: str | None,
        queue_ahead: int,
        wards: List[Dict[str, object]],
        hospitals: List[Dict[str, object]],
        route_id: str | None = None,
        include_bed_hint: bool = False,
    ) -> Dict[str, object]:
        normalized_risk = _normalize_risk_level(risk_level)
        normalized_department = str(department).strip() or "General Medicine"
        normalized_preferred_hospital = str(preferred_hospital_id).strip() or None
        normalized_queue_ahead = max(0, int(queue_ahead))
        normalized_priority = max(1.0, min(10.0, float(priority_score)))
        normalized_route_id = str(route_id).strip() or str(uuid.uuid4())

        if not wards:
            raise ValueError("No wards configured for routing")

        candidate_wards = wards
        if normalized_preferred_hospital:
            preferred_wards = [
                ward
                for ward in wards
                if str(ward.get("hospital_id", "")) == normalized_preferred_hospital
            ]
            if preferred_wards:
                candidate_wards = preferred_wards

        if not candidate_wards:
            raise ValueError("No wards configured for routing")

        hospital_map = {str(item["hospital_id"]): item for item in hospitals}
        weights = _routing_weights(normalized_risk, normalized_priority)

        hospitals_with_capacity = {
            str(ward["hospital_id"])
            for ward in candidate_wards
            if int(ward.get("available_beds", 0)) > 0
        }
        min_capacity_hospital_load = 1.0
        if hospitals_with_capacity:
            min_capacity_hospital_load = min(
                float(hospital_map.get(hospital_id, {}).get("load_ratio", 1.0))
                for hospital_id in hospitals_with_capacity
            )

        candidates: List[Dict[str, object]] = []
        for ward in candidate_wards:
            hospital_id = str(ward["hospital_id"])
            hospital_snapshot = hospital_map.get(hospital_id, {})

            available_beds = int(ward.get("available_beds", 0))
            ward_load_ratio = float(ward.get("load_ratio", 1.0))
            hospital_load_ratio = float(hospital_snapshot.get("load_ratio", ward_load_ratio))
            has_capacity = available_beds > 0
            has_other_capacity = any(
                other_hospital_id != hospital_id for other_hospital_id in hospitals_with_capacity
            )

            specialty_score, specialty_tier = _specialty_match_score(
                normalized_department,
                str(ward.get("specialty", "")),
            )
            availability_score = 1.0 if has_capacity else 0.0
            ward_load_score = max(0.0, 1.0 - ward_load_ratio)
            hospital_balance_score = max(0.0, 1.0 - hospital_load_ratio)

            preferred_boost = (
                0.03
                if normalized_preferred_hospital and hospital_id == normalized_preferred_hospital
                else 0.0
            )
            transfer_penalty = _transfer_penalty(
                risk_level=normalized_risk,
                preferred_hospital_id=normalized_preferred_hospital,
                candidate_hospital_id=hospital_id,
            )
            balancing_penalty, hard_balance_block = _balance_penalty(
                risk_level=normalized_risk,
                hospital_load_ratio=hospital_load_ratio,
                min_capacity_hospital_load=min_capacity_hospital_load,
                has_other_capacity=has_other_capacity,
            )

            acuity_boost = 0.0
            if normalized_risk == "High":
                acuity_boost += 0.06
                if specialty_tier is not None and specialty_tier <= 1:
                    acuity_boost += 0.03
            elif normalized_risk == "Medium" and specialty_tier == 0:
                acuity_boost += 0.02

            base_score = (
                (availability_score * weights["availability"])
                + (specialty_score * weights["specialty"])
                + (ward_load_score * weights["ward_load"])
                + (hospital_balance_score * weights["hospital_load"])
            )

            composite_score = base_score + acuity_boost + preferred_boost
            composite_score -= transfer_penalty
            composite_score -= balancing_penalty
            if not has_capacity:
                composite_score -= 0.18
            if hard_balance_block:
                composite_score -= 0.25

            policy_flags: List[str] = []
            if specialty_tier is not None and specialty_tier > 0:
                policy_flags.append(f"specialty_fallback_tier_{specialty_tier + 1}")
            if transfer_penalty > 0:
                policy_flags.append("transfer_penalty_applied")
            if balancing_penalty > 0:
                policy_flags.append("hospital_balancing_penalty_applied")
            if hard_balance_block:
                policy_flags.append("hospital_hard_balance_constraint")

            candidates.append(
                {
                    "hospital_id": hospital_id,
                    "hospital_name": str(ward.get("hospital_name", hospital_id)),
                    "ward_id": str(ward.get("ward_id")),
                    "ward_name": str(ward.get("ward_name", ward.get("ward_id", ""))),
                    "specialty": str(ward.get("specialty", "General Medicine")),
                    "available_beds": available_beds,
                    "capacity": int(ward.get("capacity", 0)),
                    "load_status": str(ward.get("load_status", "OK")),
                    "ward_load_ratio": round(ward_load_ratio, 4),
                    "hospital_load_ratio": round(hospital_load_ratio, 4),
                    "has_capacity": has_capacity,
                    "specialty_tier": specialty_tier,
                    "transfer_penalty": round(transfer_penalty, 4),
                    "balancing_penalty": round(balancing_penalty, 4),
                    "hard_balance_block": hard_balance_block,
                    "policy_flags": policy_flags,
                    "composite_score": round(composite_score, 6),
                    "component_scores": {
                        "availability": round(availability_score, 4),
                        "specialty_match": round(specialty_score, 4),
                        "ward_load": round(ward_load_score, 4),
                        "hospital_balance": round(hospital_balance_score, 4),
                        "acuity_boost": round(acuity_boost, 4),
                        "preferred_hospital_boost": round(preferred_boost, 4),
                        "transfer_penalty": round(transfer_penalty, 4),
                        "balancing_penalty": round(balancing_penalty, 4),
                    },
                }
            )

        candidates.sort(
            key=lambda item: (
                0 if bool(item["has_capacity"]) else 1,
                0 if not bool(item["hard_balance_block"]) else 1,
                -float(item["composite_score"]),
                int(item["specialty_tier"] if item["specialty_tier"] is not None else 99),
                -int(item["available_beds"]),
                float(item["ward_load_ratio"]),
                float(item["hospital_load_ratio"]),
                str(item["hospital_id"]),
                str(item["ward_id"]),
            )
        )

        best = candidates[0]
        no_capacity_anywhere = not any(bool(item["has_capacity"]) for item in candidates)

        wait_estimate = self._estimate_wait_for_snapshot(
            ward={
                "available_beds": best["available_beds"],
                "load_ratio": best["ward_load_ratio"],
            },
            queue_ahead=normalized_queue_ahead if bool(best["has_capacity"]) else normalized_queue_ahead + 1,
        )

        bed_hint = None
        if include_bed_hint and bool(best["has_capacity"]):
            bed_hint = self.occupancy.peek_available_bed(str(best["ward_id"]))

        overflow_risk = "none"
        if no_capacity_anywhere:
            overflow_risk = "critical"
        elif not bool(best["has_capacity"]) or bool(best["hard_balance_block"]):
            overflow_risk = "warning"
        elif str(best["load_status"]) == "Critical":
            overflow_risk = "warning"
        elif str(best["load_status"]) == "Warning":
            overflow_risk = "low"

        alternatives = [
            {
                "hospital_id": str(item["hospital_id"]),
                "hospital_name": str(item["hospital_name"]),
                "ward_id": str(item["ward_id"]),
                "ward_name": str(item["ward_name"]),
                "has_capacity": bool(item["has_capacity"]),
                "available_beds": int(item["available_beds"]),
                "load_status": str(item["load_status"]),
                "specialty_tier": item["specialty_tier"],
                "transfer_penalty": float(item["transfer_penalty"]),
                "balancing_penalty": float(item["balancing_penalty"]),
                "policy_flags": item["policy_flags"],
                "composite_score": float(item["composite_score"]),
            }
            for item in candidates[:3]
        ]

        return {
            "route_id": normalized_route_id,
            "patient_id": str(patient_id).strip() or "unknown",
            "risk_level": normalized_risk,
            "priority_score": round(normalized_priority, 2),
            "department": normalized_department,
            "preferred_hospital_id": normalized_preferred_hospital,
            "queue_ahead": normalized_queue_ahead,
            "recommended_hospital_id": str(best["hospital_id"]),
            "recommended_hospital_name": str(best["hospital_name"]),
            "recommended_ward_id": str(best["ward_id"]),
            "recommended_ward_name": str(best["ward_name"]),
            "recommended_bed_id": str(bed_hint["bed_id"]) if bed_hint else None,
            "recommended_bed_label": str(bed_hint["bed_label"]) if bed_hint else None,
            "estimated_wait_minutes": int(wait_estimate["estimated_wait_minutes"]),
            "estimated_wait_band": str(wait_estimate["wait_band"]),
            "has_capacity": bool(best["has_capacity"]),
            "overflow_risk": overflow_risk,
            "specialty_match_tier": int(best["specialty_tier"]) + 1 if best["specialty_tier"] is not None else None,
            "route_reason": "Selected best ward using specialty fallback tiers, transfer penalties, and hospital load balancing constraints",
            "explanation_fields": {
                "weights": weights,
                "policy": {
                    "specialty_fallback_tiers": _fallback_tiers(normalized_department),
                    "transfer_penalty_enabled": bool(normalized_preferred_hospital),
                    "balancing_constraint_reference_load": round(min_capacity_hospital_load, 4),
                },
                "selected_candidate": {
                    "composite_score": float(best["composite_score"]),
                    "specialty_tier": best["specialty_tier"],
                    "transfer_penalty": best["transfer_penalty"],
                    "balancing_penalty": best["balancing_penalty"],
                    "policy_flags": best["policy_flags"],
                    "component_scores": best["component_scores"],
                },
            },
            "alternatives": alternatives,
        }

    def route_patient(
        self,
        patient_id: str,
        risk_level: str,
        priority_score: float,
        department: str,
        preferred_hospital_id: str | None = None,
        queue_ahead: int = 0,
    ) -> Dict[str, object]:
        wards = self._clone_ward_snapshots(self.occupancy.get_ward_occupancy())
        hospitals = self._build_hospital_snapshots(wards)
        return self._route_patient_from_snapshots(
            patient_id=patient_id,
            risk_level=risk_level,
            priority_score=priority_score,
            department=department,
            preferred_hospital_id=preferred_hospital_id,
            queue_ahead=queue_ahead,
            wards=wards,
            hospitals=hospitals,
            route_id=str(uuid.uuid4()),
            include_bed_hint=True,
        )

    def distribute_patient_inflow(self, patients: List[Dict[str, object]]) -> Dict[str, object]:
        if not patients:
            raise ValueError("patients list cannot be empty")
        if len(patients) > 200:
            raise ValueError("patients list cannot exceed 200 records")

        virtual_wards = self._clone_ward_snapshots(self.occupancy.get_ward_occupancy())
        if not virtual_wards:
            raise ValueError("No wards configured for routing")

        def normalized_priority(payload: Dict[str, object]) -> float:
            try:
                return max(1.0, min(10.0, float(payload.get("priority_score", 1.0))))
            except (TypeError, ValueError):
                return 1.0

        prioritized = sorted(
            enumerate(patients),
            key=lambda item: (
                -normalized_priority(item[1]),
                -_risk_rank(str(item[1].get("risk_level", "Low"))),
                self._resolve_queue_ahead(item[1]),
                item[0],
            ),
        )

        assignments: List[Dict[str, object]] = []
        for inflow_rank, (source_queue_index, payload) in enumerate(prioritized, start=1):
            queue_ahead = self._resolve_queue_ahead(payload)
            hospitals = self._build_hospital_snapshots(virtual_wards)

            routing = self._route_patient_from_snapshots(
                patient_id=str(payload.get("patient_id", "")).strip() or f"inflow-{source_queue_index + 1}",
                risk_level=str(payload.get("risk_level", "Low")),
                priority_score=normalized_priority(payload),
                department=str(payload.get("department", "General Medicine")),
                preferred_hospital_id=(
                    str(payload.get("preferred_hospital_id", "")).strip() or None
                ),
                queue_ahead=queue_ahead,
                wards=virtual_wards,
                hospitals=hospitals,
                route_id=str(uuid.uuid4()),
                include_bed_hint=False,
            )

            routing["inflow_rank"] = inflow_rank
            routing["source_queue_index"] = source_queue_index
            assignments.append(routing)

            if bool(routing.get("has_capacity")):
                self._consume_virtual_capacity(
                    wards=virtual_wards,
                    ward_id=str(routing.get("recommended_ward_id", "")),
                )

        projected_hospitals = self._build_hospital_snapshots(virtual_wards)

        hospital_name_map = {
            str(item["hospital_id"]): str(item["hospital_name"])
            for item in projected_hospitals
        }
        ward_name_map = {
            str(item["ward_id"]): {
                "ward_name": str(item["ward_name"]),
                "hospital_id": str(item["hospital_id"]),
                "hospital_name": str(item["hospital_name"]),
            }
            for item in virtual_wards
        }

        hospital_counter = Counter(str(item["recommended_hospital_id"]) for item in assignments)
        ward_counter = Counter(str(item["recommended_ward_id"]) for item in assignments)

        total_capacity = int(sum(int(item["capacity"]) for item in virtual_wards))
        total_occupied = int(sum(int(item["occupied_beds"]) for item in virtual_wards))
        network_load_ratio = 0.0 if total_capacity == 0 else float(total_occupied) / float(total_capacity)

        return {
            "total_incoming_requests": len(patients),
            "ordering_policy": "priority_score_desc_then_risk_level_then_queue_ahead",
            "served_with_capacity": sum(1 for item in assignments if bool(item.get("has_capacity"))),
            "overflow_recommended": sum(
                1
                for item in assignments
                if str(item.get("overflow_risk", "none")) in {"warning", "critical"}
            ),
            "assignments": assignments,
            "hospital_distribution": [
                {
                    "hospital_id": hospital_id,
                    "hospital_name": hospital_name_map.get(hospital_id, hospital_id),
                    "assigned_patients": int(count),
                }
                for hospital_id, count in sorted(
                    hospital_counter.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "ward_distribution": [
                {
                    "ward_id": ward_id,
                    "ward_name": ward_name_map.get(ward_id, {}).get("ward_name", ward_id),
                    "hospital_id": ward_name_map.get(ward_id, {}).get("hospital_id", ""),
                    "hospital_name": ward_name_map.get(ward_id, {}).get("hospital_name", ""),
                    "assigned_patients": int(count),
                }
                for ward_id, count in sorted(
                    ward_counter.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "projected_summary": {
                "ward_count": len(virtual_wards),
                "hospital_count": len(projected_hospitals),
                "total_capacity": total_capacity,
                "total_occupied": total_occupied,
                "total_available": max(0, total_capacity - total_occupied),
                "network_load_ratio": round(network_load_ratio, 4),
                "network_load_percent": round(network_load_ratio * 100.0, 1),
            },
            "projected_hospitals": projected_hospitals,
            "projected_wards": sorted(
                virtual_wards,
                key=lambda item: (
                    str(item.get("hospital_name", "")),
                    str(item.get("ward_name", "")),
                ),
            ),
        }
