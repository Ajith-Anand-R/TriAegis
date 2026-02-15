// TriAegis TypeScript types

export interface PatientPayload {
    Patient_ID: string;
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
    };
    charts?: {
        risk_counts: Record<string, number>;
        dept_counts: Record<string, number>;
        trend: { date: string; risk_level: string; count: number }[];
        priority_categories: Record<string, number>;
        symptoms: Record<string, number>;
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
