"use client";

import { useEffect, useState, useCallback } from "react";
import {
    getQueue,
    callNextPatient,
    updateQueueStatus,
    clearCompleted,
    adminDeleteRecent,
    adminDeleteSpecific,
    getCurrentUser,
} from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PriorityBadge, StatusBadge } from "@/components/risk-badge";
import { MetricCard } from "@/components/metric-card";
import {
    ClipboardList,
    UserRoundCheck,
    Play,
    CheckCircle,
    Trash2,
    RefreshCw,
    AlertTriangle,
    Clock,
    Loader2,
    Bell,
    ShieldAlert,
    DatabaseZap,
    Trash,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";

export default function QueuePage() {
    const { t } = useI18n();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [queueData, setQueueData] = useState<any>(null);
    const [filter, setFilter] = useState("all");
    const [loading, setLoading] = useState(true);
    const [actioning, setActioning] = useState<number | null>(null);
    const [role, setRole] = useState<string | null>(null);
    const [adminScope, setAdminScope] = useState<"all" | "queue" | "predictions" | "patients">("all");
    const [specificType, setSpecificType] = useState<"patient" | "prediction" | "queue">("patient");
    const [specificValue, setSpecificValue] = useState("");
    const [adminBusy, setAdminBusy] = useState(false);
    const [adminMessage, setAdminMessage] = useState<string | null>(null);

    const fetchQueue = useCallback(async () => {
        setLoading(true);
        try {
            const data = await getQueue(filter);
            setQueueData(data);
        } catch {
            /* ignore */
        } finally {
            setLoading(false);
        }
    }, [filter]);

    useEffect(() => { fetchQueue(); }, [fetchQueue]);

    useEffect(() => {
        getCurrentUser().then((me) => setRole(me.role)).catch(() => setRole(null));
    }, []);

    const handleCallNext = async () => {
        try {
            await callNextPatient();
            fetchQueue();
        } catch (e: unknown) {
            alert(e instanceof Error ? e.message : t("simulation.noPatientsWaiting"));
        }
    };

    const handleStatusChange = async (queueId: number, status: string) => {
        setActioning(queueId);
        try {
            await updateQueueStatus(queueId, status);
            fetchQueue();
        } finally {
            setActioning(null);
        }
    };

    const handleClear = async () => {
        await clearCompleted();
        fetchQueue();
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
            fetchQueue();
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
            setSpecificValue("");
            fetchQueue();
        } catch (error: unknown) {
            setAdminMessage(error instanceof Error ? error.message : "Delete failed");
        } finally {
            setAdminBusy(false);
        }
    };

    const queue = queueData?.queue || [];
    const alerts = queueData?.alerts || [];

    return (
        <div className="space-y-6">
            <Card className="border-border/50 bg-card p-5">
                <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                        <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
                            <ClipboardList className="h-6 w-6 text-primary" />
                            Queue Command Center
                        </h1>
                        <p className="mt-1 text-sm text-muted-foreground">
                            Operational queue board with live triage controls.
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <Button variant="outline" size="sm" onClick={fetchQueue}>
                            <RefreshCw className="mr-1 h-3 w-3" /> {t("queue.refresh")}
                        </Button>
                        <Button
                            onClick={handleCallNext}
                            size="sm"
                            className="bg-primary text-primary-foreground"
                        >
                            <UserRoundCheck className="mr-1 h-3 w-3" /> {t("queue.callNext")}
                        </Button>
                        <Button variant="outline" size="sm" onClick={handleClear}>
                            <Trash2 className="mr-1 h-3 w-3" /> {t("queue.clearCompleted")}
                        </Button>
                    </div>
                </div>
            </Card>

            {role === "Admin" ? (
                <Card className="border-border/50 bg-card p-5">
                    <div className="mb-4 flex items-center gap-2">
                        <ShieldAlert className="h-4 w-4 text-primary" />
                        <h2 className="text-sm font-semibold">Admin Data Controls</h2>
                    </div>
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                        <div className="space-y-2 rounded-lg border border-border/50 p-4">
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
                                    {adminBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash className="h-3 w-3" />}
                                </Button>
                            </div>
                        </div>
                        <div className="space-y-2 rounded-lg border border-border/50 p-4">
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
                                    {adminBusy ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <DatabaseZap className="mr-1 h-3 w-3" />}
                                    Purge 30d
                                </Button>
                            </div>
                        </div>
                    </div>
                    {adminMessage ? (
                        <p className="mt-3 text-xs text-muted-foreground">{adminMessage}</p>
                    ) : null}
                </Card>
            ) : null}

            {alerts.length > 0 && (
                <div className="space-y-2">
                    {alerts.map((alertItem: string, index: number) => (
                        <div key={index} className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
                            <Bell className="h-4 w-4 shrink-0" /> {alertItem}
                        </div>
                    ))}
                </div>
            )}

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <MetricCard title={t("queue.waiting")} value={queueData?.waiting_count ?? 0} icon={Clock} variant="warning" />
                <MetricCard title={t("queue.critical")} value={queueData?.critical_count ?? 0} icon={AlertTriangle} variant="danger" />
                <MetricCard title={t("queue.total")} value={queue.length} icon={ClipboardList} variant="info" />
            </div>

            <Card className="border-border/50 bg-card p-3">
                <div className="flex flex-wrap items-center gap-2">
                    {["all", "waiting", "in_progress", "completed"].map((status) => (
                        <Button
                            key={status}
                            size="sm"
                            variant={filter === status ? "default" : "outline"}
                            onClick={() => setFilter(status)}
                        >
                            {status === "all" ? t("queue.all") : status === "in_progress" ? t("queue.inProgress") : status === "completed" ? t("queue.completed") : t("queue.waiting")}
                        </Button>
                    ))}
                </div>
            </Card>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {loading ? (
                    <Card className="col-span-full flex h-40 items-center justify-center border-border/50 bg-card">
                        <Loader2 className="h-6 w-6 animate-spin text-primary" />
                    </Card>
                ) : queue.length === 0 ? (
                    <Card className="col-span-full flex h-40 flex-col items-center justify-center border-border/50 bg-card text-muted-foreground">
                        <ClipboardList className="mb-2 h-8 w-8 opacity-50" />
                        {t("queue.empty")}
                    </Card>
                ) : (
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    queue.map((item: any, index: number) => (
                        <Card key={item.queue_id} className="border-border/50 bg-card p-4">
                            <div className="mb-3 flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-muted-foreground">#{index + 1} • Queue ID {item.queue_id}</p>
                                    <p className="font-mono text-sm font-semibold">{item.patient_id || `P-${item.prediction_id}`}</p>
                                </div>
                                <StatusBadge status={item.status} />
                            </div>

                            <div className="mb-3 flex flex-wrap items-center gap-2">
                                <PriorityBadge category={item.priority_score >= 9 ? "Critical" : item.priority_score >= 7 ? "High" : item.priority_score >= 5 ? "Standard" : "Low"} />
                                <span className="text-xs text-muted-foreground">Score: {Number(item.priority_score).toFixed(1)}</span>
                                <span className="text-xs text-muted-foreground">Dept: {item.department}</span>
                            </div>

                            <div className="text-xs text-muted-foreground">
                                <p>Wait: {item.dynamic_estimated_wait || item.estimated_wait_time}</p>
                                {item.monitoring_note ? (
                                    <p className={item.deteriorating ? "text-red-400" : "text-muted-foreground"}>{item.monitoring_note}</p>
                                ) : null}
                            </div>

                            <div className="mt-3 flex justify-end gap-1">
                                {item.status === "waiting" && (
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        disabled={actioning === item.queue_id}
                                        onClick={() => handleStatusChange(item.queue_id, "in_progress")}
                                    >
                                        {actioning === item.queue_id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
                                    </Button>
                                )}
                                {item.status === "in_progress" && (
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        disabled={actioning === item.queue_id}
                                        onClick={() => handleStatusChange(item.queue_id, "completed")}
                                    >
                                        {actioning === item.queue_id ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle className="h-3 w-3" />}
                                    </Button>
                                )}
                            </div>
                        </Card>
                    ))
                )}
            </div>
        </div>
    );
}
