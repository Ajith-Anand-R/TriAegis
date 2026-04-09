// TriAegis TypeScript types

export interface PatientPayload {
    Patient_ID: string;
    "Patient Name"?: string;
    Age: number;
    Gender: string;
    Symptoms: string;
    "Blood Pressure": string;
    "Heart Rate": number;
    Temperature: number;
    "Pre-Existing Conditions": string;
}

export interface Prediction {
    patient_id: string;
    risk_level: "Low" | "Medium" | "High";
    confidence: number;
    probabilities: { Low: number; Medium: number; High: number };
    priority_score: number;
    priority_category: string;
    department: string;
    department_reason: string;
    urgency: string;
    estimated_wait_time: string;
    routing_logic: string;
    queue_position: number;
    processing_time_ms: number;
    esi_level?: number;
    out_of_distribution?: boolean;
    ood_score?: number;
    ood_threshold?: number;
    manual_review_required?: boolean;
    differential_diagnosis?: Array<{
        diagnosis: string;
        likelihood: string;
        rationale: string;
    }>;
}

export interface ShapContributor {
    feature: string;
    feature_key: string;
    impact: number;
    direction: string;
    value: number | string;
    interpretation: string;
    rank: number;
}

export interface Explanation {
    prediction: {
        risk_level: string;
        confidence: number;
        probabilities: { Low: number; Medium: number; High: number };
    };
    explanation: {
        base_risk: number;
        final_risk: number;
        top_contributors: ShapContributor[];
        protective_factors: ShapContributor[];
        confidence_factors: string[];
    };
}

export interface PredictResponse {
    patient: PatientPayload;
    prediction: Prediction;
    explanation: Explanation;
    saved?: boolean;
    prediction_id?: number;
    confidence_band?: string;
    clinical_explanation?: string;
    manual_review_recommended?: boolean;
}

export interface BatchResponse {
    source_rows: Record<string, unknown>[];
    source_count: number;
    results: Prediction[];
    risk_counts: {
        total: number;
        high: number;
        medium: number;
        low: number;
    };
}

export interface QueueItem {
    queue_id: number;
    prediction_id: number;
    arrival_time: string;
    priority_score: number;
    department: string;
    status: string;
    queue_position: number;
    estimated_wait_time: string;
    patient_id?: string;
    patient_name?: string;
}

export interface QueueResponse {
    queue: QueueItem[];
    waiting_count: number;
    critical_count: number;
    alerts: string[];
    departments: string[];
}

export interface DashboardSummary {
    total_predictions: number;
    queue_waiting: number;
    high_risk_cases: number;
    active_departments: number;
}

export interface AnalyticsData {
    empty: boolean;
    metrics?: {
        total_predictions: number;
        high_risk_rate: number;
        avg_priority: number;
        avg_confidence: number;
        queue_waiting: number;
        critical_waiting: number;
        recent_high_risk: number;
        routing_total_routes?: number;
        routing_capacity_hit_rate?: number;
        routing_overflow_rate?: number;
        routing_mean_wait_delta?: number;
        routing_admit_conversion_rate?: number;
    };
    charts?: {
        risk_counts: Record<string, number>;
        dept_counts: Record<string, number>;
        trend: { date: string; risk_level: string; count: number }[];
        priority_categories: Record<string, number>;
        symptoms: Record<string, number>;
    };
    routing_quality?: {
        window_hours: number;
        total_routes: number;
        capacity_hit_count: number;
        capacity_hit_rate: number;
        overflow_count: number;
        overflow_rate: number;
        admitted_count: number;
        admit_conversion_rate: number;
        mean_estimated_wait_minutes: number;
        mean_actual_wait_minutes: number;
        mean_wait_delta_minutes: number;
    };
    recent_activity?: Record<string, unknown>[];
    filter_options?: {
        risk_levels: string[];
        departments: string[];
        date_range: { min: string; max: string };
    };
}

export interface HistoryResponse {
    empty: boolean;
    records: Record<string, unknown>[];
    filter_options: {
        departments?: string[];
    };
}

export interface HealthcheckData {
    ok: boolean;
    checks: Record<string, string>;
    details: Record<string, unknown>;
}

export interface RouteDecision {
    route_id: string;
    patient_id: string;
    risk_level: string;
    priority_score: number;
    department: string;
    preferred_hospital_id?: string | null;
    queue_ahead?: number;
    inflow_rank?: number;
    source_queue_index?: number;
    recommended_hospital_id: string;
    recommended_hospital_name: string;
    recommended_ward_id: string;
    recommended_ward_name: string;
    recommended_bed_id?: string | null;
    recommended_bed_label?: string | null;
    estimated_wait_minutes: number;
    estimated_wait_band: string;
    has_capacity: boolean;
    overflow_risk: string;
    specialty_match_tier?: number | null;
    route_reason: string;
    explanation_fields: Record<string, unknown>;
    alternatives: RouteAlternative[];
}

export interface RouteAlternative {
    hospital_id: string;
    hospital_name: string;
    ward_id: string;
    ward_name: string;
    has_capacity: boolean;
    available_beds: number;
    load_status: string;
    specialty_tier?: number | null;
    transfer_penalty?: number;
    balancing_penalty?: number;
    policy_flags?: string[];
    composite_score: number;
}

export interface RouteTrackingSummary {
    route_id: string;
    patient_id: string;
    admitted?: boolean;
    admitted_bed_id?: string | null;
    actual_wait_minutes?: number;
    wait_delta_minutes?: number;
    created_at?: string;
    admitted_at?: string;
    discharged?: boolean;
    discharged_at?: string;
}

export interface RoutingMetrics {
    window_hours: number;
    total_routes: number;
    capacity_hit_count: number;
    capacity_hit_rate: number;
    overflow_count: number;
    overflow_rate: number;
    admitted_count: number;
    admit_conversion_rate: number;
    mean_estimated_wait_minutes: number;
    mean_actual_wait_minutes: number;
    mean_wait_delta_minutes: number;
}

export interface RouteResponse {
    routing: RouteDecision;
    routing_metrics?: RoutingMetrics;
    summary: Record<string, unknown>;
}

export interface RouteAdmitResponse extends RouteResponse {
    admitted: boolean;
    admission: Record<string, unknown> | null;
    admit_error: string | null;
    route_tracking: RouteTrackingSummary | null;
}

export interface RouteDistributionPatientPayload {
    patient_id?: string;
    risk_level: string;
    priority_score: number;
    department: string;
    preferred_hospital_id?: string;
    queue_position?: number;
    queue_ahead?: number;
}

export interface RouteDistributionResult {
    total_incoming_requests: number;
    ordering_policy: string;
    served_with_capacity: number;
    overflow_recommended: number;
    assignments: RouteDecision[];
    hospital_distribution: Array<{
        hospital_id: string;
        hospital_name: string;
        assigned_patients: number;
    }>;
    ward_distribution: Array<{
        ward_id: string;
        ward_name: string;
        hospital_id: string;
        hospital_name: string;
        assigned_patients: number;
    }>;
    projected_summary: {
        ward_count: number;
        hospital_count: number;
        total_capacity: number;
        total_occupied: number;
        total_available: number;
        network_load_ratio: number;
        network_load_percent: number;
    };
    projected_hospitals: Record<string, unknown>[];
    projected_wards: Record<string, unknown>[];
}

export interface RouteDistributeResponse {
    distribution: RouteDistributionResult;
    persisted_routes: number;
    route_ids: string[];
    routing_metrics?: RoutingMetrics;
    summary: Record<string, unknown>;
}
