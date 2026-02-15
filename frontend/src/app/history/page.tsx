"use client";

import { useEffect, useState, useCallback, Fragment } from "react";
import { getHistory, clearOldRecords, adminDeleteRecent, adminDeleteSpecific, getCurrentUser } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RiskBadge, PriorityBadge } from "@/components/risk-badge";
import {
    History,
    Search,
    Loader2,
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

    const removeRecordsFromUi = useCallback((predicate: (record: any) => boolean) => {
        setData((prev: any) => {
            if (!prev || !Array.isArray(prev.records)) {
                return prev;
            }
            return {
                ...prev,
                records: prev.records.filter((record: any) => !predicate(record)),
            };
        });
    }, []);

    useEffect(() => {
        fetchData();
        getCurrentUser().then((me) => setRole(me.role)).catch(() => setRole(null));
        const timer = window.setInterval(() => {
            fetchData(true);
        }, 10000);
        return () => window.clearInterval(timer);
    }, [fetchData]);

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

    const records = data?.records || [];
    const departments = data?.filter_options?.departments || [];

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
                        <History className="h-6 w-6 text-primary" />
                        {t("history.title")}
                    </h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        {t("history.subtitle")}
                    </p>
                </div>
                <Button variant="ghost" size="sm" onClick={handlePurge} className="text-muted-foreground hover:text-destructive">
                    <Trash2 className="mr-1 h-3 w-3" /> {t("history.purge")}
                </Button>
            </div>

            {role === "Admin" ? (
                <Card className="border-border/50 bg-card p-4">
                    <div className="mb-2 text-sm font-semibold">Admin Data Controls</div>
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                        <div className="space-y-2 rounded-lg border border-border/50 p-3">
                            <Label className="text-xs text-muted-foreground">Delete specific record</Label>
                            <div className="flex gap-2">
                                <select
                                    value={specificType}
                                    onChange={(event) => setSpecificType(event.target.value as "patient" | "prediction" | "queue")}
                                    aria-label="Delete specific record type"
                                    title="Delete specific record type"
                                    className="h-10 rounded-md border border-input bg-input/50 px-3 text-sm"
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
                                />
                                <Button size="sm" onClick={handleDeleteSpecific} disabled={adminBusy}>
                                    {adminBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                                </Button>
                            </div>
                        </div>

                        <div className="space-y-2 rounded-lg border border-border/50 p-3">
                            <Label className="text-xs text-muted-foreground">Delete last 30 days data</Label>
                            <div className="flex gap-2">
                                <select
                                    value={adminScope}
                                    onChange={(event) => setAdminScope(event.target.value as "all" | "queue" | "predictions" | "patients")}
                                    aria-label="Delete last 30 days scope"
                                    title="Delete last 30 days scope"
                                    className="h-10 rounded-md border border-input bg-input/50 px-3 text-sm"
                                >
                                    <option value="all">All datasets</option>
                                    <option value="queue">Queue only</option>
                                    <option value="predictions">History only</option>
                                    <option value="patients">Users only</option>
                                </select>
                                <Button size="sm" variant="outline" onClick={handleDeleteRecent30Days} disabled={adminBusy}>
                                    {adminBusy ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
                                    Purge 30d
                                </Button>
                            </div>
                        </div>
                    </div>
                    {adminMessage ? <p className="mt-2 text-xs text-muted-foreground">{adminMessage}</p> : null}
                </Card>
            ) : null}

            {/* Filters */}
            <Card className="border-border/50 bg-card p-4">
                <div className="flex flex-wrap items-end gap-4">
                    <div className="min-w-[200px] flex-1">
                        <Label className="text-xs text-muted-foreground mb-1.5">{t("history.searchPatient")}</Label>
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
                    <div>
                        <Label className="text-xs text-muted-foreground mb-1.5">{t("history.riskLevel")}</Label>
                        <div className="flex gap-1.5">
                            {["High", "Medium", "Low"].map((rl) => (
                                <Button
                                    key={rl}
                                    size="sm"
                                    variant={riskFilter.includes(rl) ? "default" : "outline"}
                                    onClick={() => setRiskFilter((prev) =>
                                        prev.includes(rl) ? prev.filter((r) => r !== rl) : [...prev, rl]
                                    )}
                                    className={riskFilter.includes(rl) ? "bg-primary/20 text-primary" : "text-muted-foreground"}
                                >
                                    {rl}
                                </Button>
                            ))}
                        </div>
                    </div>
                    {departments.length > 0 && (
                        <div>
                            <Label htmlFor="history-department" className="text-xs text-muted-foreground mb-1.5">{t("history.department")}</Label>
                            <select
                                id="history-department"
                                aria-label={t("history.department")}
                                title={t("history.department")}
                                value={deptFilter}
                                onChange={(e) => setDeptFilter(e.target.value)}
                                className="flex h-9 w-44 rounded-md border border-input bg-input/50 px-3 text-sm"
                            >
                                <option value="">{t("history.allDepartments")}</option>
                                {departments.map((d: string) => (
                                    <option key={d} value={d}>{d}</option>
                                ))}
                            </select>
                        </div>
                    )}
                    <div>
                        <Label className="text-xs text-muted-foreground mb-1.5">{t("history.from")}</Label>
                        <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-36 bg-input/50" />
                    </div>
                    <div>
                        <Label className="text-xs text-muted-foreground mb-1.5">{t("history.to")}</Label>
                        <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="w-36 bg-input/50" />
                    </div>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => { setSearchId(""); setRiskFilter([]); setDeptFilter(""); setStartDate(""); setEndDate(""); }}
                    >
                        {t("analytics.reset")}
                    </Button>
                </div>
            </Card>

            {/* Results */}
            <Card className="border-border/50 bg-card overflow-hidden">
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
                        <div className="max-h-[600px] overflow-auto">
                            <table className="w-full text-sm">
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
                                                onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                                            >
                                                <td className="px-4 py-2.5 font-mono text-xs">{row.patient_id}</td>
                                                <td className="px-4 py-2.5 text-xs text-muted-foreground">
                                                    {row.timestamp?.split("T")[0] || row.timestamp?.split(" ")[0]}
                                                </td>
                                                <td className="px-4 py-2.5"><RiskBadge level={row.risk_level} size="sm" /></td>
                                                <td className="px-4 py-2.5"><PriorityBadge category={row.priority_category} /></td>
                                                <td className="px-4 py-2.5 font-semibold">{Number(row.priority_score).toFixed(1)}</td>
                                                <td className="px-4 py-2.5 text-muted-foreground">{row.recommended_department}</td>
                                                <td className="px-4 py-2.5 text-muted-foreground">{(Number(row.model_confidence) * 100).toFixed(0)}%</td>
                                                {role === "Admin" ? (
                                                    <td className="px-4 py-2.5 text-right">
                                                        <Button
                                                            size="sm"
                                                            variant="ghost"
                                                            className="h-7 px-2 text-destructive hover:text-destructive"
                                                            title="Delete this record"
                                                            onClick={async (event) => {
                                                                event.stopPropagation();
                                                                const predictionId = Number(row.prediction_id);
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
                                                            }}
                                                            disabled={adminBusy}
                                                        >
                                                            {adminBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                                                        </Button>
                                                    </td>
                                                ) : null}
                                                <td className="px-2 py-2.5">
                                                    {expandedRow === i ? <ChevronUp className="h-3.5 w-3.5 text-muted-foreground" /> : <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />}
                                                </td>
                                            </tr>
                                            {expandedRow === i && (
                                                <tr key={`detail-${i}`}>
                                                    <td colSpan={role === "Admin" ? 9 : 8} className="bg-muted/10 px-6 py-4">
                                                        <div className="grid grid-cols-2 gap-4 text-xs md:grid-cols-4">
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
                                                    </td>
                                                </tr>
                                            )}
                                        </Fragment>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </Card>
        </div>
    );
}
