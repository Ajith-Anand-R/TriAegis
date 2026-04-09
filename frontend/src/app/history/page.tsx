"use client";

import { useEffect, useState, useCallback, useMemo, Fragment, type KeyboardEvent } from "react";
import { getHistory, clearOldRecords, adminDeleteRecent, adminDeleteAllData, adminDeleteSpecific, getCurrentUser, submitOutcomeFeedback, getOutcomeFeedback } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RiskBadge, PriorityBadge } from "@/components/risk-badge";
import {
    CalendarDays,
    DatabaseZap,
    History,
    Search,
    Loader2,
    RefreshCw,
    SlidersHorizontal,
    Trash2,
    ChevronDown,
    ChevronUp,
    FileText,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";

export default function HistoryPage() {
    const { t } = useI18n();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [searchId, setSearchId] = useState("");
    const [riskFilter, setRiskFilter] = useState<string[]>([]);
    const [deptFilter, setDeptFilter] = useState("");
    const [startDate, setStartDate] = useState("");
    const [endDate, setEndDate] = useState("");
    const [expandedRow, setExpandedRow] = useState<number | null>(null);
    const [role, setRole] = useState<string | null>(null);
    const [adminScope, setAdminScope] = useState<"all" | "queue" | "predictions" | "patients">("all");
    const [specificType, setSpecificType] = useState<"patient" | "prediction" | "queue">("patient");
    const [specificValue, setSpecificValue] = useState("");
    const [adminBusy, setAdminBusy] = useState(false);
    const [adminMessage, setAdminMessage] = useState<string | null>(null);
    const [feedbackBusy, setFeedbackBusy] = useState<number | null>(null);
    const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
    const [feedbackRisk, setFeedbackRisk] = useState<"Low" | "Medium" | "High">("Medium");
    const [feedbackDepartment, setFeedbackDepartment] = useState("General Medicine");
    const [feedbackOutcome, setFeedbackOutcome] = useState("discharged");
    const [feedbackNotes, setFeedbackNotes] = useState("");
    const [feedbackStats, setFeedbackStats] = useState<{ count: number; agreement_rate: number } | null>(null);

    const fetchData = useCallback(async (background = false) => {
        if (!background) {
            setLoading(true);
        }
        try {
            const filters: Record<string, string> = {};
            if (searchId) filters.patient_id = searchId;
            if (riskFilter.length) filters.risk_levels = riskFilter.join(",");
            if (deptFilter) filters.departments = deptFilter;
            if (startDate) filters.start_date = startDate;
            if (endDate) filters.end_date = endDate;
            const d = await getHistory(filters);
            setData(d);
        } catch {
            /* ignore */
        } finally {
            if (!background) {
                setLoading(false);
            }
        }
    }, [searchId, riskFilter, deptFilter, startDate, endDate]);

    const removeRecordsFromUi = useCallback((predicate: (record: Record<string, unknown>) => boolean) => {
        setData((prev: unknown) => {
            if (!prev || typeof prev !== "object") {
                return prev;
            }

            const prevWithRecords = prev as { records?: unknown } & Record<string, unknown>;
            if (!Array.isArray(prevWithRecords.records)) {
                return prev;
            }

            return {
                ...prevWithRecords,
                records: prevWithRecords.records.filter((record) => predicate(record as Record<string, unknown>) === false),
            };
        });
    }, []);

    const toggleExpandedRow = useCallback((rowIndex: number) => {
        setExpandedRow((prev) => (prev === rowIndex ? null : rowIndex));
    }, []);

    const handleHistoryRowKeyDown = useCallback((event: KeyboardEvent<HTMLTableRowElement>, rowIndex: number) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            toggleExpandedRow(rowIndex);
        }
    }, [toggleExpandedRow]);

    useEffect(() => {
        fetchData();
        getCurrentUser().then((me) => setRole(me.role)).catch(() => setRole(null));
        getOutcomeFeedback(50)
            .then((rows) => {
                setFeedbackStats({
                    count: Number(rows.count || 0),
                    agreement_rate: Number(rows.agreement_rate || 0),
                });
            })
            .catch(() => setFeedbackStats(null));
        const timer = window.setInterval(() => {
            fetchData(true);
        }, 10000);
        return () => window.clearInterval(timer);
    }, [fetchData]);

    const submitFeedbackForPrediction = async (predictionId: number) => {
        setFeedbackBusy(predictionId);
        setFeedbackMessage(null);
        try {
            await submitOutcomeFeedback({
                prediction_id: predictionId,
                actual_risk_level: feedbackRisk,
                final_department: feedbackDepartment,
                outcome_status: feedbackOutcome,
                clinician_role: role || "Doctor",
                notes: feedbackNotes,
            });
            const updated = await getOutcomeFeedback(50);
            setFeedbackStats({
                count: Number(updated.count || 0),
                agreement_rate: Number(updated.agreement_rate || 0),
            });
            setFeedbackMessage(`Outcome feedback saved for prediction ${predictionId}`);
            setFeedbackNotes("");
        } catch (error: unknown) {
            setFeedbackMessage(error instanceof Error ? error.message : "Failed to save feedback");
        } finally {
            setFeedbackBusy(null);
        }
    };

    const handlePurge = async () => {
        if (!confirm(t("history.confirmPurge"))) return;
        await clearOldRecords(90);
        fetchData(true);
    };

    const handleDeleteRecent30Days = async () => {
        if (role !== "Admin") {
            return;
        }
        const target = adminScope === "all" ? "all datasets" : adminScope;
        if (!window.confirm(`Delete last 30 days of ${target}?`)) {
            return;
        }

        setAdminBusy(true);
        setAdminMessage(null);
        try {
            const result = await adminDeleteRecent(30, adminScope);
            const queueDeleted = Number(result.deleted_queue_rows ?? 0);
            const predictionDeleted = Number(result.deleted_prediction_rows ?? 0);
            const patientDeleted = Number(result.deleted_patient_rows ?? 0);
            setAdminMessage(`Deleted: queue ${queueDeleted}, predictions ${predictionDeleted}, patients ${patientDeleted}`);
            await fetchData(true);
        } catch (error: unknown) {
            setAdminMessage(error instanceof Error ? error.message : "Delete failed");
        } finally {
            setAdminBusy(false);
        }
    };

    const handleDeleteSpecific = async () => {
        if (role !== "Admin") {
            return;
        }
        const raw = specificValue.trim();
        if (!raw) {
            setAdminMessage("Enter a value to delete");
            return;
        }

        if (!window.confirm(`Delete ${specificType} record: ${raw}?`)) {
            return;
        }

        setAdminBusy(true);
        setAdminMessage(null);
        try {
            const payload: { patient_id?: string; prediction_id?: number; queue_id?: number } = {};
            if (specificType === "patient") {
                payload.patient_id = raw;
            }
            if (specificType === "prediction") {
                const predictionId = Number(raw);
                if (Number.isNaN(predictionId)) {
                    throw new Error("Prediction ID must be a number");
                }
                payload.prediction_id = predictionId;
            }
            if (specificType === "queue") {
                const queueId = Number(raw);
                if (Number.isNaN(queueId)) {
                    throw new Error("Queue ID must be a number");
                }
                payload.queue_id = queueId;
            }

            const result = await adminDeleteSpecific(payload);
            const queueDeleted = Number(result.deleted_queue_rows ?? 0);
            const predictionDeleted = Number(result.deleted_prediction_rows ?? 0);
            const patientDeleted = Number(result.deleted_patient_rows ?? 0);
            setAdminMessage(`Deleted: queue ${queueDeleted}, predictions ${predictionDeleted}, patients ${patientDeleted}`);

            if (payload.prediction_id !== undefined) {
                removeRecordsFromUi((record) => Number(record.prediction_id) === payload.prediction_id);
            }
            if (payload.patient_id !== undefined) {
                removeRecordsFromUi((record) => String(record.patient_id) === payload.patient_id);
            }

            setSpecificValue("");
            await fetchData(true);
        } catch (error: unknown) {
            setAdminMessage(error instanceof Error ? error.message : "Delete failed");
        } finally {
            setAdminBusy(false);
        }
    };

    const handleDeleteAllData = async () => {
        if (role !== "Admin") {
            return;
        }

        const confirmed = window.confirm(
            "Delete ALL operational data (queue, predictions, patients, route history, and monitoring snapshots)? This cannot be undone.",
        );
        if (!confirmed) {
            return;
        }

        setAdminBusy(true);
        setAdminMessage(null);
        try {
            const result = await adminDeleteAllData();
            const queueDeleted = Number(result.deleted_queue_rows ?? 0);
            const predictionDeleted = Number(result.deleted_prediction_rows ?? 0);
            const patientDeleted = Number(result.deleted_patient_rows ?? 0);
            const routesDeleted = Number(result.deleted_route_rows ?? 0);
            setAdminMessage(
                `All data cleared: queue ${queueDeleted}, predictions ${predictionDeleted}, patients ${patientDeleted}, routes ${routesDeleted}`,
            );
            await fetchData(true);
        } catch (error: unknown) {
            setAdminMessage(error instanceof Error ? error.message : "Delete all data failed");
        } finally {
            setAdminBusy(false);
        }
    };

    const handleDeletePredictionRecord = async (predictionId: number) => {
        if (role !== "Admin") {
            return;
        }

        if (!predictionId || Number.isNaN(predictionId)) {
            setAdminMessage("Prediction ID not found for this row");
            return;
        }

        if (!window.confirm(`Delete prediction record ${predictionId}?`)) {
            return;
        }

        setAdminBusy(true);
        setAdminMessage(null);
        try {
            const result = await adminDeleteSpecific({ prediction_id: predictionId });
            const queueDeleted = Number(result.deleted_queue_rows ?? 0);
            const predictionDeleted = Number(result.deleted_prediction_rows ?? 0);
            const patientDeleted = Number(result.deleted_patient_rows ?? 0);
            setAdminMessage(`Deleted: queue ${queueDeleted}, predictions ${predictionDeleted}, patients ${patientDeleted}`);
            removeRecordsFromUi((record) => Number(record.prediction_id) === predictionId);
            await fetchData(true);
        } catch (error: unknown) {
            setAdminMessage(error instanceof Error ? error.message : "Delete failed");
        } finally {
            setAdminBusy(false);
        }
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const renderExpandedContent = (row: any) => (
        <>
            <div className="grid grid-cols-1 gap-4 text-xs sm:grid-cols-2 md:grid-cols-4">
                <div>
                    <span className="text-muted-foreground">{t("history.symptoms")}</span>
                    <p className="mt-0.5 font-medium">{row.symptoms || "—"}</p>
                </div>
                <div>
                    <span className="text-muted-foreground">{t("history.bloodPressure")}</span>
                    <p className="mt-0.5 font-medium">{row.bp_systolic}/{row.bp_diastolic}</p>
                </div>
                <div>
                    <span className="text-muted-foreground">{t("history.heartRate")}</span>
                    <p className="mt-0.5 font-medium">{row.heart_rate} bpm</p>
                </div>
                <div>
                    <span className="text-muted-foreground">{t("history.temperature")}</span>
                    <p className="mt-0.5 font-medium">{row.temperature}°F</p>
                </div>
                <div>
                    <span className="text-muted-foreground">{t("history.conditions")}</span>
                    <p className="mt-0.5 font-medium">{row.pre_existing_conditions || "—"}</p>
                </div>
                <div>
                    <span className="text-muted-foreground">{t("history.source")}</span>
                    <p className="mt-0.5 font-medium">{row.source}</p>
                </div>
                <div>
                    <span className="text-muted-foreground">{t("history.waitTime")}</span>
                    <p className="mt-0.5 font-medium">{row.estimated_wait_time}</p>
                </div>
                <div>
                    <span className="text-muted-foreground">{t("history.processing")}</span>
                    <p className="mt-0.5 font-medium">{row.processing_time_ms}ms</p>
                </div>
            </div>

            {(role === "Doctor" || role === "Admin") ? (
                <div className="mt-4 rounded-lg border border-border/50 bg-card p-3">
                    <p className="text-xs font-semibold">Clinical Outcome Feedback</p>
                    <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-4">
                        <select
                            value={feedbackRisk}
                            onChange={(event) => setFeedbackRisk(event.target.value as "Low" | "Medium" | "High")}
                            className="h-10 rounded-md border border-input bg-input/50 px-3 text-xs sm:h-9"
                            aria-label="Actual risk level"
                            title="Actual risk level"
                        >
                            <option value="Low">Low</option>
                            <option value="Medium">Medium</option>
                            <option value="High">High</option>
                        </select>
                        <Input
                            value={feedbackDepartment}
                            onChange={(event) => setFeedbackDepartment(event.target.value)}
                            className="h-10 bg-input/50 text-xs sm:h-9"
                            placeholder="Final department"
                            aria-label="Final department"
                        />
                        <select
                            value={feedbackOutcome}
                            onChange={(event) => setFeedbackOutcome(event.target.value)}
                            className="h-10 rounded-md border border-input bg-input/50 px-3 text-xs sm:h-9"
                            aria-label="Outcome status"
                            title="Outcome status"
                        >
                            <option value="discharged">discharged</option>
                            <option value="admitted">admitted</option>
                            <option value="transferred">transferred</option>
                            <option value="deceased">deceased</option>
                        </select>
                        <Button
                            size="sm"
                            disabled={feedbackBusy === Number(row.prediction_id)}
                            onClick={() => submitFeedbackForPrediction(Number(row.prediction_id))}
                            className="min-h-10 px-3 sm:min-h-9"
                        >
                            {feedbackBusy === Number(row.prediction_id) ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Save outcome"}
                        </Button>
                    </div>
                    <Input
                        value={feedbackNotes}
                        onChange={(event) => setFeedbackNotes(event.target.value)}
                        className="mt-2 h-10 bg-input/50 text-xs sm:h-9"
                        placeholder="Outcome notes"
                        aria-label="Outcome notes"
                    />
                </div>
            ) : null}
        </>
    );

    const records = useMemo(() => {
        const rows = Array.isArray(data?.records)
            ? (data.records as Record<string, unknown>[])
            : [];
        const seenPatientIds = new Set<string>();
        return rows.filter((row: Record<string, unknown>) => {
            const patientId = String(row.patient_id ?? "").trim();
            if (!patientId) {
                return true;
            }
            if (seenPatientIds.has(patientId)) {
                return false;
            }
            seenPatientIds.add(patientId);
            return true;
        });
    }, [data?.records]);
    const departments = data?.filter_options?.departments || [];
    const liveStatus = adminMessage || feedbackMessage || "";

    return (
        <div className="space-y-7">
            <p role="status" aria-live="polite" className="sr-only">{liveStatus}</p>
            <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-slate-500/15 via-card to-cyan-500/10 p-4 sm:p-6">
                <div className="pointer-events-none absolute -right-14 -top-12 h-40 w-40 rounded-full bg-slate-500/20 blur-2xl" />
                <div className="pointer-events-none absolute -bottom-16 -left-8 h-44 w-44 rounded-full bg-cyan-400/15 blur-2xl" />

                <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div>
                        <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-cyan-300">
                            Clinical Audit Trail
                        </p>
                        <h1 className="font-display mt-1 flex items-center gap-2 text-2xl font-semibold tracking-tight md:text-3xl">
                            <History className="h-6 w-6 text-primary" />
                            {t("history.title")}
                        </h1>
                        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
                            {t("history.subtitle")}
                        </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                            Records: {records.length}
                        </span>
                        <Button variant="outline" size="sm" className="min-h-10 border-primary/30 px-3 sm:min-h-9" onClick={() => fetchData(true)}>
                            <RefreshCw className="mr-1 h-3.5 w-3.5" />
                            Refresh
                        </Button>
                        <Button variant="ghost" size="sm" onClick={handlePurge} className="min-h-10 px-3 text-muted-foreground hover:text-destructive sm:min-h-9">
                            <Trash2 className="mr-1 h-3 w-3" /> {t("history.purge")}
                        </Button>
                    </div>
                </div>
            </section>

            {role === "Admin" ? (
                <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm">
                    <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
                        <DatabaseZap className="h-4 w-4 text-primary" />
                        Admin Data Controls
                    </div>
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                        <div className="space-y-2 rounded-lg border border-border/50 bg-muted/20 p-3">
                            <Label className="text-xs text-muted-foreground">Delete specific record</Label>
                            <div className="flex flex-col gap-2 sm:flex-row">
                                <select
                                    value={specificType}
                                    onChange={(event) => setSpecificType(event.target.value as "patient" | "prediction" | "queue")}
                                    aria-label="Delete specific record type"
                                    title="Delete specific record type"
                                    className="h-10 w-full rounded-md border border-input bg-input/50 px-3 text-sm sm:w-auto"
                                >
                                    <option value="patient">Patient ID</option>
                                    <option value="prediction">Prediction ID</option>
                                    <option value="queue">Queue ID</option>
                                </select>
                                <Input
                                    value={specificValue}
                                    onChange={(event) => setSpecificValue(event.target.value)}
                                    placeholder={specificType === "patient" ? "e.g. P001" : "e.g. 12"}
                                    className="bg-input/50"
                                    aria-label="Delete specific record value"
                                />
                                <Button size="sm" onClick={handleDeleteSpecific} disabled={adminBusy} className="min-h-10 w-full px-3 sm:min-h-9 sm:w-auto" aria-label="Delete specific record">
                                    {adminBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                                </Button>
                            </div>
                        </div>

                        <div className="space-y-2 rounded-lg border border-border/50 bg-muted/20 p-3">
                            <Label className="text-xs text-muted-foreground">Delete last 30 days data</Label>
                            <div className="flex flex-col gap-2 sm:flex-row">
                                <select
                                    value={adminScope}
                                    onChange={(event) => setAdminScope(event.target.value as "all" | "queue" | "predictions" | "patients")}
                                    aria-label="Delete last 30 days scope"
                                    title="Delete last 30 days scope"
                                    className="h-10 w-full rounded-md border border-input bg-input/50 px-3 text-sm sm:w-auto"
                                >
                                    <option value="all">All datasets</option>
                                    <option value="queue">Queue only</option>
                                    <option value="predictions">History only</option>
                                    <option value="patients">Patients only</option>
                                </select>
                                <Button size="sm" variant="outline" onClick={handleDeleteRecent30Days} disabled={adminBusy} className="min-h-10 w-full px-3 sm:min-h-9 sm:w-auto">
                                    {adminBusy ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
                                    Delete last 30 days
                                </Button>
                            </div>
                        </div>
                    </div>
                    <div className="mt-4 rounded-lg border border-destructive/30 bg-destructive/10 p-3">
                        <Label className="text-xs text-muted-foreground">Emergency reset</Label>
                        <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                            <p className="text-xs text-muted-foreground">
                                Use only when you need to wipe all operational patient data.
                            </p>
                            <Button
                                size="sm"
                                variant="destructive"
                                onClick={handleDeleteAllData}
                                disabled={adminBusy}
                                className="min-h-10 w-full px-3 sm:min-h-9 sm:w-auto"
                            >
                                {adminBusy ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Trash2 className="mr-1 h-3 w-3" />}
                                Clear all data
                            </Button>
                        </div>
                    </div>
                    {adminMessage ? <p className="mt-2 text-xs text-muted-foreground" role="status" aria-live="polite">{adminMessage}</p> : null}
                </Card>
            ) : null}

            {feedbackStats ? (
                <Card className="border-border/60 bg-gradient-to-r from-card to-card/90 p-4 shadow-sm">
                    <div className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
                        <p>Outcome feedback records: <span className="font-semibold text-primary">{feedbackStats.count}</span></p>
                        <p>Predicted vs actual agreement: <span className="font-semibold text-emerald-300">{feedbackStats.agreement_rate.toFixed(1)}%</span></p>
                    </div>
                    {feedbackMessage ? <p className="mt-2 text-xs text-muted-foreground" role="status" aria-live="polite">{feedbackMessage}</p> : null}
                </Card>
            ) : null}

            {/* Filters */}
            <Card className="border-border/60 bg-gradient-to-r from-card to-card/90 p-4 shadow-sm">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                        <SlidersHorizontal className="h-4 w-4 text-primary" />
                        <p className="font-display text-sm font-semibold">Filter and Search</p>
                    </div>
                    <Button
                        variant="ghost"
                        size="sm"
                        className="min-h-10 px-3 sm:min-h-9"
                        onClick={() => { setSearchId(""); setRiskFilter([]); setDeptFilter(""); setStartDate(""); setEndDate(""); }}
                    >
                        {t("analytics.reset")}
                    </Button>
                </div>

                <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
                    <div className="min-w-0 lg:col-span-4">
                        <Label className="mb-1.5 block text-xs text-muted-foreground">{t("history.searchPatient")}</Label>
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                            <Input
                                placeholder={t("history.searchPlaceholder")}
                                value={searchId}
                                onChange={(e) => setSearchId(e.target.value)}
                                className="bg-input/50 pl-9"
                            />
                        </div>
                    </div>

                    <div className="lg:col-span-3">
                        <Label className="mb-1.5 block text-xs text-muted-foreground">{t("history.riskLevel")}</Label>
                        <div className="flex flex-wrap gap-1.5">
                            {["High", "Medium", "Low"].map((rl) => (
                                <Button
                                    key={rl}
                                    size="sm"
                                    variant={riskFilter.includes(rl) ? "default" : "outline"}
                                    onClick={() => setRiskFilter((prev) =>
                                        prev.includes(rl) ? prev.filter((r) => r !== rl) : [...prev, rl]
                                    )}
                                    className={`${riskFilter.includes(rl) ? "bg-primary/20 text-primary" : "text-muted-foreground"} min-h-10 px-3 sm:min-h-9`}
                                >
                                    {rl}
                                </Button>
                            ))}
                        </div>
                    </div>

                    {departments.length > 0 && (
                        <div className="lg:col-span-2">
                            <Label htmlFor="history-department" className="mb-1.5 block text-xs text-muted-foreground">{t("history.department")}</Label>
                            <select
                                id="history-department"
                                aria-label={t("history.department")}
                                title={t("history.department")}
                                value={deptFilter}
                                onChange={(e) => setDeptFilter(e.target.value)}
                                className="flex h-9 w-full rounded-md border border-input bg-input/50 px-3 text-sm"
                            >
                                <option value="">{t("history.allDepartments")}</option>
                                {departments.map((d: string) => (
                                    <option key={d} value={d}>{d}</option>
                                ))}
                            </select>
                        </div>
                    )}

                    <div className="lg:col-span-1">
                        <Label className="mb-1.5 flex items-center gap-1 text-xs text-muted-foreground">
                            <CalendarDays className="h-3 w-3" />
                            {t("history.from")}
                        </Label>
                        <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="h-10 w-full bg-input/50 sm:h-9" />
                    </div>

                    <div className="lg:col-span-1">
                        <Label className="mb-1.5 flex items-center gap-1 text-xs text-muted-foreground">
                            <CalendarDays className="h-3 w-3" />
                            {t("history.to")}
                        </Label>
                        <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="h-10 w-full bg-input/50 sm:h-9" />
                    </div>
                </div>
            </Card>

            {/* Results */}
            <Card className="overflow-hidden border-border/60 bg-gradient-to-b from-card to-card/90 shadow-sm">
                {loading ? (
                    <div className="flex h-48 items-center justify-center">
                        <Loader2 className="h-6 w-6 animate-spin text-primary" />
                    </div>
                ) : records.length === 0 ? (
                    <div className="flex h-48 flex-col items-center justify-center text-center text-muted-foreground">
                        <FileText className="mb-2 h-8 w-8 opacity-40" />
                        <p className="font-medium">{t("history.noRecords")}</p>
                        <p className="text-xs">{t("history.adjustFilters")}</p>
                    </div>
                ) : (
                    <div>
                        <div className="border-b border-border/50 bg-muted/20 px-4 py-2 text-xs text-muted-foreground">
                            {records.length} {t("history.recordsFound")}
                        </div>
                        <div className="space-y-3 p-3 md:hidden">
                            {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                            {records.map((row: any, i: number) => {
                                const predictionId = Number(row.prediction_id);
                                const expanded = expandedRow === i;

                                return (
                                    <article key={`history-mobile-${row.prediction_id || i}`} className="rounded-xl border border-border/60 bg-card/70 p-3">
                                        <div className="flex items-start justify-between gap-2">
                                            <div>
                                                <p className="font-mono text-xs font-semibold">{row.patient_id}</p>
                                                {row.patient_name ? (
                                                    <p className="text-[11px] text-muted-foreground">{row.patient_name}</p>
                                                ) : null}
                                                <p className="text-[11px] text-muted-foreground">
                                                    {row.timestamp?.split("T")[0] || row.timestamp?.split(" ")[0]}
                                                </p>
                                            </div>
                                            {role === "Admin" ? (
                                                <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    className="h-8 px-2 text-destructive hover:text-destructive"
                                                    title="Delete this record"
                                                    aria-label={`Delete prediction ${String(row.prediction_id || "record")}`}
                                                    onClick={() => handleDeletePredictionRecord(predictionId)}
                                                    disabled={adminBusy}
                                                >
                                                    {adminBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                                                </Button>
                                            ) : null}
                                        </div>

                                        <div className="mt-2 flex flex-wrap items-center gap-2">
                                            <RiskBadge level={row.risk_level} size="sm" />
                                            <PriorityBadge category={row.priority_category} />
                                            <span className="text-xs text-muted-foreground">{t("common.score")}: {Number(row.priority_score).toFixed(1)}</span>
                                        </div>

                                        <p className="mt-2 text-xs text-muted-foreground">{t("common.department")}: {row.recommended_department}</p>
                                        <p className="text-xs text-muted-foreground">{t("history.confidence")}: {(Number(row.model_confidence) * 100).toFixed(0)}%</p>

                                        <Button
                                            type="button"
                                            variant="ghost"
                                            size="sm"
                                            className="mt-2 min-h-9 w-full justify-between px-3 text-xs"
                                            onClick={() => toggleExpandedRow(i)}
                                            aria-expanded={expanded}
                                        >
                                            {expanded ? "Hide details" : "Show details"}
                                            {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                                        </Button>

                                        {expanded ? (
                                            <div className="mt-3 rounded-lg bg-muted/20 p-3">
                                                {renderExpandedContent(row)}
                                            </div>
                                        ) : null}
                                    </article>
                                );
                            })}
                        </div>

                        <div className="hidden md:block">
                            <div className="mobile-scroll max-h-[600px] overflow-auto">
                                <table className="w-full min-w-[980px] text-sm">
                                    <caption className="sr-only">Prediction history records table</caption>
                                    <thead className="sticky top-0 bg-muted/80 backdrop-blur">
                                        <tr className="border-b border-border/50">
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">{t("history.patient")}</th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">{t("history.date")}</th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">{t("common.risk")}</th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">{t("common.priority")}</th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">{t("common.score")}</th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">{t("common.department")}</th>
                                            <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">{t("history.confidence")}</th>
                                            {role === "Admin" ? <th className="px-4 py-2.5 text-right text-xs font-medium text-muted-foreground">{t("common.actions")}</th> : null}
                                            <th className="w-8"></th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                        {records.map((row: any, i: number) => (
                                            <Fragment key={`row-group-${row.prediction_id || i}`}>
                                                <tr
                                                    key={`row-${i}`}
                                                    className="border-b border-border/30 cursor-pointer transition-colors hover:bg-muted/20"
                                                    onClick={() => toggleExpandedRow(i)}
                                                    onKeyDown={(event) => handleHistoryRowKeyDown(event, i)}
                                                    tabIndex={0}
                                                    role="button"
                                                    aria-expanded={expandedRow === i}
                                                >
                                                    <td className="px-4 py-3 text-xs">
                                                        <p className="font-mono font-semibold">{row.patient_id}</p>
                                                        {row.patient_name ? (
                                                            <p className="text-[11px] text-muted-foreground">{row.patient_name}</p>
                                                        ) : null}
                                                    </td>
                                                    <td className="px-4 py-3 text-xs text-muted-foreground">
                                                        {row.timestamp?.split("T")[0] || row.timestamp?.split(" ")[0]}
                                                    </td>
                                                    <td className="px-4 py-3"><RiskBadge level={row.risk_level} size="sm" /></td>
                                                    <td className="px-4 py-3"><PriorityBadge category={row.priority_category} /></td>
                                                    <td className="px-4 py-3 font-semibold">{Number(row.priority_score).toFixed(1)}</td>
                                                    <td className="px-4 py-3 text-muted-foreground">{row.recommended_department}</td>
                                                    <td className="px-4 py-3 text-muted-foreground">{(Number(row.model_confidence) * 100).toFixed(0)}%</td>
                                                    {role === "Admin" ? (
                                                        <td className="px-4 py-3 text-right">
                                                            <Button
                                                                size="sm"
                                                                variant="ghost"
                                                                className="h-7 px-2 text-destructive hover:text-destructive"
                                                                title="Delete this record"
                                                                aria-label={`Delete prediction ${String(row.prediction_id || "record")}`}
                                                                onClick={(event) => {
                                                                    event.stopPropagation();
                                                                    handleDeletePredictionRecord(Number(row.prediction_id));
                                                                }}
                                                                disabled={adminBusy}
                                                            >
                                                                {adminBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                                                            </Button>
                                                        </td>
                                                    ) : null}
                                                    <td className="px-2 py-3">
                                                        {expandedRow === i ? <ChevronUp className="h-3.5 w-3.5 text-muted-foreground" /> : <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />}
                                                    </td>
                                                </tr>
                                                {expandedRow === i && (
                                                    <tr key={`detail-${i}`}>
                                                        <td colSpan={role === "Admin" ? 9 : 8} className="bg-muted/10 px-6 py-4">
                                                            {renderExpandedContent(row)}
                                                        </td>
                                                    </tr>
                                                )}
                                            </Fragment>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}
            </Card>
        </div>
    );
}
