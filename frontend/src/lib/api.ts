// API client for TriAegis FastAPI backend

import type {
    RouteAdmitResponse,
    RouteDistributeResponse,
    RouteDistributionPatientPayload,
    RouteResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const AUTH_TOKEN_KEY = "triaegis_access_token";
const AUTH_ROLE_KEY = "triaegis_user_role";

let authTokenMemory: string | null = null;

function getStoredToken(): string | null {
    if (typeof window !== "undefined") {
        return window.localStorage.getItem(AUTH_TOKEN_KEY);
    }
    return authTokenMemory;
}

function buildAuthHeaders(extraHeaders?: HeadersInit): Headers {
    const headers = new Headers(extraHeaders);
    const token = getStoredToken();
    if (token) {
        headers.set("Authorization", `Bearer ${token}`);
    }
    return headers;
}

export function setAuthToken(token: string): void {
    authTokenMemory = token;
    if (typeof window !== "undefined") {
        window.localStorage.setItem(AUTH_TOKEN_KEY, token);
    }
}

export function getAuthToken(): string | null {
    return getStoredToken();
}

export function clearAuthToken(): void {
    authTokenMemory = null;
    if (typeof window !== "undefined") {
        window.localStorage.removeItem(AUTH_TOKEN_KEY);
        window.localStorage.removeItem(AUTH_ROLE_KEY);
    }
}

export function setAuthUserRole(role: string): void {
    if (typeof window !== "undefined") {
        window.localStorage.setItem(AUTH_ROLE_KEY, role);
    }
}

export function getAuthUserRole(): string | null {
    if (typeof window !== "undefined") {
        return window.localStorage.getItem(AUTH_ROLE_KEY);
    }
    return null;
}

export async function login(username: string, password: string) {
    const data = await request<{ access_token: string; token_type: string }>("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
    });
    if (data.access_token) {
        setAuthToken(data.access_token);
    }
    return data;
}

export async function getCurrentUser() {
    return request<{ username: string; role: string }>("/api/auth/me");
}

export async function getAuthUsers() {
    return request<{ users: Array<{ username: string; role: string; is_active: boolean; created_at: string; updated_at: string }> }>("/api/auth/users");
}

export async function registerUser(payload: { username: string; password: string; role: string }) {
    return request<{ created: { username: string; role: string; is_active: boolean } }>("/api/auth/register", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function registerUserWithAdminCredentials(payload: {
    admin_username: string;
    admin_password: string;
    username: string;
    password: string;
    role: string;
}) {
    return request<{ created: { username: string; role: string; is_active: boolean } }>("/api/auth/register-by-admin", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function changePassword(payload: { current_password: string; new_password: string }) {
    return request<{ message: string }>("/api/auth/change-password", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateUserStatus(payload: { username: string; is_active: boolean }) {
    return request<{ updated: { username: string; role: string; is_active: boolean } }>("/api/auth/users/status", {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

function getErrorDetail(payload: unknown, fallback: string): string {
    if (payload && typeof payload === "object" && "detail" in payload) {
        const detail = (payload as { detail?: unknown }).detail;
        if (typeof detail === "string") {
            return detail;
        }
    }
    return fallback;
}

function handleAuthError(status: number): void {
    if (status !== 401) {
        return;
    }

    clearAuthToken();
    if (typeof window !== "undefined" && window.location.pathname !== "/login") {
        window.location.href = "/login";
    }
}

function isMethodNotAllowedError(error: unknown): boolean {
    if (!(error instanceof Error)) {
        return false;
    }
    return /method not allowed|405/i.test(error.message);
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
    const headers = buildAuthHeaders(options?.headers);
    if (!headers.has("Content-Type") && !(options?.body instanceof FormData)) {
        headers.set("Content-Type", "application/json");
    }

    const res = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers,
        cache: "no-store",
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const detail = getErrorDetail(err, res.statusText);
        handleAuthError(res.status);
        throw new Error(detail || JSON.stringify(err));
    }
    return res.json();
}

// Constants
export async function getConstants() {
    return request<{ symptoms: string[]; conditions: string[] }>("/api/constants");
}

// Dashboard
export async function getDashboard() {
    return request<{
        total_predictions: number;
        queue_waiting: number;
        high_risk_cases: number;
        active_departments: number;
    }>("/api/dashboard");
}

// Predict
export async function predictSingle(patient: Record<string, unknown>) {
    return request<Record<string, unknown>>("/api/predict", {
        method: "POST",
        body: JSON.stringify(patient),
    });
}

export async function predictAndSave(patient: Record<string, unknown>) {
    return request<Record<string, unknown>>("/api/predict/save", {
        method: "POST",
        body: JSON.stringify(patient),
    });
}

export async function routePatient(payload: {
    patient_id?: string;
    risk_level: string;
    priority_score: number;
    department: string;
    preferred_hospital_id?: string;
    queue_ahead?: number;
}) {
    return request<RouteResponse>("/api/route", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function routeAndAdmit(payload: {
    patient_id?: string;
    risk_level: string;
    priority_score: number;
    department: string;
    preferred_hospital_id?: string;
    queue_ahead?: number;
}) {
    return request<RouteAdmitResponse>("/api/route/admit", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function distributeRouteInflow(payload: {
    patients: RouteDistributionPatientPayload[];
    persist_routes?: boolean;
}) {
    return request<RouteDistributeResponse>("/api/route/distribute", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function getSymptomDialogue(payload: {
    presenting_complaint: string;
    transcript?: Array<{ role: string; content: string }>;
    known_patient?: Record<string, unknown>;
    model?: string;
    max_followup_questions?: number;
}) {
    return request<Record<string, unknown>>("/api/triage/symptom-dialogue", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function submitOutcomeFeedback(payload: {
    prediction_id: number;
    actual_risk_level: string;
    final_department: string;
    outcome_status?: string;
    clinician_role?: string;
    notes?: string;
}) {
    return request<Record<string, unknown>>("/api/feedback/outcome", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function getOutcomeFeedback(limit = 200) {
    return request<Record<string, unknown>>(`/api/feedback/outcome?limit=${limit}`);
}

export async function predictBatch(file: File) {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API_BASE}/api/predict/batch`, {
        method: "POST",
        headers: buildAuthHeaders(),
        body: form,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const detail = getErrorDetail(err, res.statusText);
        handleAuthError(res.status);
        throw new Error(detail || JSON.stringify(err));
    }
    return res.json();
}

// Queue
export async function getQueue(status?: string) {
    const params = status ? `?status=${status}` : "";
    return request<Record<string, unknown>>(`/api/queue${params}`);
}

export async function callNextPatient() {
    return request<Record<string, unknown>>("/api/queue/next", { method: "POST" });
}

export async function updateQueueStatus(queueId: number, status: string) {
    return request<Record<string, unknown>>(`/api/queue/${queueId}/status`, {
        method: "PATCH",
        body: JSON.stringify({ status }),
    });
}

export async function clearCompleted() {
    return request<Record<string, unknown>>("/api/queue/completed", { method: "DELETE" });
}

// Analytics
export async function getAnalytics(filters?: Record<string, string>) {
    const params = new URLSearchParams(filters || {}).toString();
    return request<Record<string, unknown>>(`/api/analytics${params ? `?${params}` : ""}`);
}

// History
export async function getHistory(filters?: Record<string, string>) {
    const params = new URLSearchParams(filters || {}).toString();
    return request<Record<string, unknown>>(`/api/history${params ? `?${params}` : ""}`);
}

export async function clearOldRecords(days = 90) {
    return request<Record<string, unknown>>(`/api/history/old?days=${days}`, { method: "DELETE" });
}

export async function adminDeleteSpecific(payload: { patient_id?: string; prediction_id?: number; queue_id?: number }) {
    const path = "/api/admin/data/specific";
    const requestWithMethod = async (method: "POST" | "DELETE") => {
        return request<Record<string, unknown>>(path, {
            method,
            body: JSON.stringify(payload),
        });
    };

    try {
        return await requestWithMethod("POST");
    } catch (error: unknown) {
        if (isMethodNotAllowedError(error)) {
            return requestWithMethod("DELETE");
        }
        throw error;
    }
}

export async function adminDeleteRecent(days = 30, scope: "all" | "queue" | "predictions" | "patients" = "all") {
    const path = `/api/admin/data/recent?days=${days}&scope=${scope}`;
    try {
        return await request<Record<string, unknown>>(path, { method: "POST" });
    } catch (error: unknown) {
        if (isMethodNotAllowedError(error)) {
            return request<Record<string, unknown>>(path, { method: "DELETE" });
        }
        throw error;
    }
}

export async function adminDeleteAllData() {
    const path = "/api/admin/data/all";
    try {
        return await request<Record<string, unknown>>(path, { method: "POST" });
    } catch (error: unknown) {
        if (isMethodNotAllowedError(error)) {
            return request<Record<string, unknown>>(path, { method: "DELETE" });
        }
        throw error;
    }
}

// Upload
export async function uploadDocument(file: File) {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        headers: buildAuthHeaders(),
        body: form,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const detail = getErrorDetail(err, res.statusText);
        handleAuthError(res.status);
        throw new Error(detail || JSON.stringify(err));
    }
    return res.json();
}

// Healthcheck
export async function getHealthcheck() {
    return request<Record<string, unknown>>("/api/healthcheck");
}

// Fairness
export async function getFairness(sampleSize = 2000) {
    return request<Record<string, unknown>>(`/api/fairness?sample_size=${sampleSize}`);
}

export async function exportFairnessCsv(reportType: string, sampleSize = 2000) {
    const headers = buildAuthHeaders();
    const res = await fetch(
        `${API_BASE}/api/fairness/export?report_type=${encodeURIComponent(reportType)}&sample_size=${sampleSize}`,
        {
            method: "GET",
            headers,
        }
    );
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const detail = getErrorDetail(err, res.statusText);
        handleAuthError(res.status);
        throw new Error(detail || JSON.stringify(err));
    }
    return res.blob();
}

// Simulation
export async function getSimulationState() {
    return request<Record<string, unknown>>("/api/simulation/state");
}

export async function resetSimulation() {
    return request<Record<string, unknown>>("/api/simulation/reset", { method: "POST" });
}

export async function stepSimulation(minutes: number, lambdaRate: number, seed: number) {
    return request<Record<string, unknown>>("/api/simulation/step", {
        method: "POST",
        body: JSON.stringify({ minutes, lambda_rate: lambdaRate, seed }),
    });
}

export async function exportSimulationCsv(reportType: "queue" | "timeline") {
    const headers = buildAuthHeaders();
    const res = await fetch(`${API_BASE}/api/simulation/export?report_type=${reportType}`, {
        method: "GET",
        headers,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        const detail = getErrorDetail(err, res.statusText);
        handleAuthError(res.status);
        throw new Error(detail || JSON.stringify(err));
    }
    return res.blob();
}

// Exports — returns blob
export async function exportPdfSingle(patient: Record<string, unknown>, prediction: Record<string, unknown>) {
    const headers = buildAuthHeaders({ "Content-Type": "application/json" });
    const res = await fetch(`${API_BASE}/api/export/pdf/single`, {
        method: "POST",
        headers,
        body: JSON.stringify({ patient, prediction }),
    });
    return res.blob();
}

export async function exportPdfBatch(results: Record<string, unknown>[]) {
    const headers = buildAuthHeaders({ "Content-Type": "application/json" });
    const res = await fetch(`${API_BASE}/api/export/pdf/batch`, {
        method: "POST",
        headers,
        body: JSON.stringify(results),
    });
    return res.blob();
}

export async function exportCsv(records: Record<string, unknown>[]) {
    const headers = buildAuthHeaders({ "Content-Type": "application/json" });
    const res = await fetch(`${API_BASE}/api/export/csv`, {
        method: "POST",
        headers,
        body: JSON.stringify(records),
    });
    return res.blob();
}
