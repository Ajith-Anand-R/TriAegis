"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import {
    getQueue,
    callNextPatient,
    updateQueueStatus,
    clearCompleted,
    adminDeleteRecent,
    adminDeleteAllData,
    adminDeleteSpecific,
    getCurrentUser,
    routeAndAdmit,
    routePatient,
    distributeRouteInflow,
} from "@/lib/api";
import type { RouteDecision, RouteDistributeResponse } from "@/lib/types";
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
    Search,
    ShieldAlert,
    DatabaseZap,
    Trash,
    Route,
    Building2,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";

export default function QueuePage() {
    const { t } = useI18n();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [queueData, setQueueData] = useState<any>(null);
    const [filter, setFilter] = useState("all");
    const [searchTerm, setSearchTerm] = useState("");
    const [loading, setLoading] = useState(true);
    const [actioning, setActioning] = useState<number | null>(null);
    const [role, setRole] = useState<string | null>(null);
    const [adminScope, setAdminScope] = useState<"all" | "queue" | "predictions" | "patients">("all");
    const [specificType, setSpecificType] = useState<"patient" | "prediction" | "queue">("patient");
    const [specificValue, setSpecificValue] = useState("");
    const [adminBusy, setAdminBusy] = useState(false);
    const [adminMessage, setAdminMessage] = useState<string | null>(null);
    const [routeLoadingQueueId, setRouteLoadingQueueId] = useState<number | null>(null);
    const [routeAdmitQueueId, setRouteAdmitQueueId] = useState<number | null>(null);
    const [routesByQueueId, setRoutesByQueueId] = useState<Record<number, RouteDecision>>({});
    const [routeMessageByQueueId, setRouteMessageByQueueId] = useState<Record<number, string>>({});
    const [routeErrorByQueueId, setRouteErrorByQueueId] = useState<Record<number, string>>({});
    const [distributionBusy, setDistributionBusy] = useState(false);
    const [distributionMessage, setDistributionMessage] = useState("");
    const [distributionError, setDistributionError] = useState("");

    const inferRiskLevel = (priorityScoreRaw: unknown, fallback: unknown): string => {
        const fallbackText = String(fallback || "").trim();
        if (fallbackText === "High" || fallbackText === "Medium" || fallbackText === "Low") {
            return fallbackText;
        }

        const priorityScore = Number(priorityScoreRaw || 0);
        if (priorityScore >= 8.5) {
            return "High";
        }
        if (priorityScore >= 5.0) {
            return "Medium";
        }
        return "Low";
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const buildRoutePayload = (item: any, index: number) => {
        const queuePosition = Number(item.queue_position || index + 1);
        return {
            patient_id: String(item.patient_id || `P-${item.prediction_id}`),
            risk_level: inferRiskLevel(item.priority_score, item.risk_level),
            priority_score: Number(item.priority_score || 1),
            department: String(item.department || "General Medicine"),
            queue_ahead: Math.max(0, queuePosition - 1),
        };
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const buildDistributionPayload = (item: any, index: number) => {
        const routePayload = buildRoutePayload(item, index);
        const queuePosition = Number(item.queue_position || index + 1);
        return {
            ...routePayload,
            queue_position: queuePosition,
        };
    };

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
            setRoutesByQueueId({});
            setRouteMessageByQueueId({});
            setRouteErrorByQueueId({});

            const queueDeleted = Number(result.deleted_queue_rows ?? 0);
            const predictionDeleted = Number(result.deleted_prediction_rows ?? 0);
            const patientDeleted = Number(result.deleted_patient_rows ?? 0);
            const routesDeleted = Number(result.deleted_route_rows ?? 0);
            setAdminMessage(
                `All data cleared: queue ${queueDeleted}, predictions ${predictionDeleted}, patients ${patientDeleted}, routes ${routesDeleted}`,
            );
            await fetchQueue();
        } catch (error: unknown) {
            setAdminMessage(error instanceof Error ? error.message : "Delete all data failed");
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

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleRecommendRoute = async (item: any, index: number) => {
        const queueId = Number(item.queue_id);
        setRouteLoadingQueueId(queueId);
        setRouteErrorByQueueId((prev) => ({ ...prev, [queueId]: "" }));
        setRouteMessageByQueueId((prev) => ({ ...prev, [queueId]: "" }));

        try {
            const response = await routePatient(buildRoutePayload(item, index));
            const routing = response.routing as RouteDecision | undefined;
            if (!routing) {
                throw new Error("Routing service did not return a recommendation");
            }

            setRoutesByQueueId((prev) => ({ ...prev, [queueId]: routing }));
            setRouteMessageByQueueId((prev) => ({
                ...prev,
                [queueId]: `Recommended ${routing.recommended_hospital_name} / ${routing.recommended_ward_name}`,
            }));
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Route recommendation failed";
            setRouteErrorByQueueId((prev) => ({ ...prev, [queueId]: message }));
        } finally {
            setRouteLoadingQueueId(null);
        }
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleRouteAndAdmit = async (item: any, index: number) => {
        const queueId = Number(item.queue_id);
        setRouteAdmitQueueId(queueId);
        setRouteErrorByQueueId((prev) => ({ ...prev, [queueId]: "" }));
        setRouteMessageByQueueId((prev) => ({ ...prev, [queueId]: "" }));

        try {
            const response = await routeAndAdmit(buildRoutePayload(item, index));
            const routing = response.routing as RouteDecision | undefined;
            if (routing) {
                setRoutesByQueueId((prev) => ({ ...prev, [queueId]: routing }));
            }

            const admitted = Boolean(response.admitted);
            if (admitted) {
                const admission = (response.admission || {}) as Record<string, unknown>;
                const bedLabel = String(admission.bed_label || admission.bed_id || "bed reserved");
                setRouteMessageByQueueId((prev) => ({
                    ...prev,
                    [queueId]: `Admitted via route workflow. Reserved ${bedLabel}.`,
                }));
            } else {
                setRouteErrorByQueueId((prev) => ({
                    ...prev,
                    [queueId]: String(response.admit_error || "Route generated but direct admission was not possible"),
                }));
            }

            await fetchQueue();
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Route-to-admit failed";
            setRouteErrorByQueueId((prev) => ({ ...prev, [queueId]: message }));
        } finally {
            setRouteAdmitQueueId(null);
        }
    };

    const handleDistributeWaiting = async () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const waitingInView = visibleQueue.filter((item: any) => String(item.status) === "waiting");
        if (waitingInView.length === 0) {
            setDistributionError("No waiting patients in the current view to distribute.");
            setDistributionMessage("");
            return;
        }

        setDistributionBusy(true);
        setDistributionError("");
        setDistributionMessage("");

        try {
            const response = await distributeRouteInflow({
                patients: waitingInView.map((item: unknown, index: number) => buildDistributionPayload(item, index)),
                persist_routes: true,
            });
            const typedResponse = response as RouteDistributeResponse;
            const assignments = Array.isArray(typedResponse.distribution?.assignments)
                ? typedResponse.distribution.assignments
                : [];

            const mappedRoutes: Record<number, RouteDecision> = {};
            const mappedMessages: Record<number, string> = {};

            assignments.forEach((assignment, assignmentIndex) => {
                let queueId: number | null = null;
                const sourceIndex = Number(assignment.source_queue_index);

                if (
                    Number.isInteger(sourceIndex)
                    && sourceIndex >= 0
                    && sourceIndex < waitingInView.length
                ) {
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    queueId = Number((waitingInView[sourceIndex] as any).queue_id);
                } else {
                    // Fall back to patient_id matching when source index is absent.
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    const patientMatch = waitingInView.find((entry: any) => {
                        const localPatientId = String(entry.patient_id || `P-${entry.prediction_id}`).trim();
                        return localPatientId && localPatientId === String(assignment.patient_id || "").trim();
                    });
                    if (patientMatch) {
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        queueId = Number((patientMatch as any).queue_id);
                    }
                }

                if (!queueId || Number.isNaN(queueId)) {
                    return;
                }

                mappedRoutes[queueId] = assignment;
                mappedMessages[queueId] = `Bulk route ${assignmentIndex + 1}: ${assignment.recommended_hospital_name} / ${assignment.recommended_ward_name}`;
            });

            setRoutesByQueueId((prev) => ({ ...prev, ...mappedRoutes }));
            setRouteMessageByQueueId((prev) => ({ ...prev, ...mappedMessages }));
            setRouteErrorByQueueId((prev) => {
                const next = { ...prev };
                Object.keys(mappedRoutes).forEach((queueIdKey) => {
                    next[Number(queueIdKey)] = "";
                });
                return next;
            });

            const totalIncoming = Number(typedResponse.distribution?.total_incoming_requests || waitingInView.length);
            const servedWithCapacity = Number(typedResponse.distribution?.served_with_capacity || 0);
            const overflowRecommended = Number(typedResponse.distribution?.overflow_recommended || 0);
            const persistedRoutes = Number(typedResponse.persisted_routes || 0);

            setDistributionMessage(
                `Distributed ${totalIncoming} patients. Capacity-backed: ${servedWithCapacity}. Overflow recommendations: ${overflowRecommended}. Persisted routes: ${persistedRoutes}.`,
            );

            await fetchQueue();
        } catch (error: unknown) {
            setDistributionError(error instanceof Error ? error.message : "Batch distribution failed");
        } finally {
            setDistributionBusy(false);
        }
    };

    const queue = useMemo(() => (Array.isArray(queueData?.queue) ? queueData.queue : []), [queueData]);
    const alerts = useMemo(() => (Array.isArray(queueData?.alerts) ? queueData.alerts : []), [queueData]);
    const liveStatus = useMemo(() => {
        const firstRouteMessage = Object.values(routeMessageByQueueId).find((message) => Boolean(message));
        const firstRouteError = Object.values(routeErrorByQueueId).find((message) => Boolean(message));
        return distributionError || firstRouteError || distributionMessage || firstRouteMessage || adminMessage || "";
    }, [routeErrorByQueueId, routeMessageByQueueId, adminMessage, distributionError, distributionMessage]);
    const visibleQueue = useMemo(() => {
        const q = searchTerm.trim().toLowerCase();
        if (!q) {
            return queue;
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return queue.filter((item: any) => {
            const patient = String(item.patient_id || `P-${item.prediction_id}` || "").toLowerCase();
            const department = String(item.department || "").toLowerCase();
            const queueId = String(item.queue_id || "").toLowerCase();
            return patient.includes(q) || department.includes(q) || queueId.includes(q);
        });
    }, [queue, searchTerm]);

    return (
        <div className="space-y-7">
            <p role="status" aria-live="polite" className="sr-only">{liveStatus}</p>
            <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-indigo-500/15 via-card to-cyan-500/10 p-4 sm:p-6">
                <div className="pointer-events-none absolute -right-16 -top-12 h-44 w-44 rounded-full bg-indigo-500/20 blur-2xl" />
                <div className="pointer-events-none absolute -bottom-16 -left-8 h-44 w-44 rounded-full bg-cyan-400/15 blur-2xl" />

                <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div>
                        <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-indigo-300">
                            Queue Operations
                        </p>
                        <h1 className="font-display mt-1 flex items-center gap-2 text-2xl font-semibold tracking-tight md:text-3xl">
                            <ClipboardList className="h-6 w-6 text-primary" />
                            Queue Command Center
                        </h1>
                        <p className="mt-2 text-sm text-muted-foreground">
                            Operational queue board with live triage controls.
                        </p>
                    </div>

                    <div className="flex flex-wrap gap-2">
                        <Button variant="outline" size="sm" onClick={fetchQueue} className="min-h-10 border-primary/30 px-3 sm:min-h-9">
                            <RefreshCw className="mr-1 h-3.5 w-3.5" /> {t("queue.refresh")}
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleDistributeWaiting}
                            disabled={distributionBusy || loading}
                            className="min-h-10 border-primary/30 px-3 sm:min-h-9"
                        >
                            {distributionBusy ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : <Route className="mr-1 h-3.5 w-3.5" />}
                            Distribute Waiting
                        </Button>
                        <Button
                            onClick={handleCallNext}
                            size="sm"
                            className="min-h-10 bg-gradient-to-r from-indigo-600 to-cyan-600 px-3 text-white shadow-lg shadow-indigo-500/20 hover:from-indigo-500 hover:to-cyan-500 sm:min-h-9"
                        >
                            <UserRoundCheck className="mr-1 h-3.5 w-3.5" /> {t("queue.callNext")}
                        </Button>
                        <Button variant="outline" size="sm" onClick={handleClear} className="min-h-10 px-3 sm:min-h-9">
                            <Trash2 className="mr-1 h-3.5 w-3.5" /> {t("queue.clearCompleted")}
                        </Button>
                    </div>
                </div>
            </section>

            {distributionError ? (
                <p className="-mt-3 text-xs text-destructive" role="alert">{distributionError}</p>
            ) : null}
            {distributionMessage ? (
                <p className="-mt-3 text-xs text-emerald-400" role="status" aria-live="polite">{distributionMessage}</p>
            ) : null}

            {role === "Admin" ? (
                <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-5 shadow-sm">
                    <div className="mb-4 flex items-center gap-2">
                        <ShieldAlert className="h-4 w-4 text-primary" />
                        <h2 className="font-display text-sm font-semibold">Admin Data Controls</h2>
                    </div>
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                        <div className="space-y-2 rounded-lg border border-border/50 bg-muted/20 p-4">
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
                                    {adminBusy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash className="h-3 w-3" />}
                                </Button>
                            </div>
                        </div>
                        <div className="space-y-2 rounded-lg border border-border/50 bg-muted/20 p-4">
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
                                    {adminBusy ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <DatabaseZap className="mr-1 h-3 w-3" />}
                                    Delete last 30 days
                                </Button>
                            </div>
                        </div>
                    </div>
                    <div className="mt-4 rounded-lg border border-destructive/30 bg-destructive/10 p-4">
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
                    {adminMessage ? (
                        <p className="mt-3 text-xs text-muted-foreground" role="status" aria-live="polite">{adminMessage}</p>
                    ) : null}
                </Card>
            ) : null}

            {alerts.length > 0 && (
                <div className="space-y-2" role="status" aria-live="polite" aria-label="Queue alerts">
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

            <Card className="border-border/60 bg-gradient-to-r from-card to-card/90 p-4 shadow-sm">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <p className="font-display text-sm font-semibold">Queue Filters</p>
                    <span className="rounded-full border border-border/60 bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground">
                        Showing {visibleQueue.length} of {queue.length}
                    </span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                    {["all", "waiting", "in_progress", "completed"].map((status) => (
                        <Button
                            key={status}
                            size="sm"
                            variant={filter === status ? "default" : "outline"}
                            onClick={() => setFilter(status)}
                            className="min-h-10 px-3 sm:min-h-9"
                            aria-pressed={filter === status}
                        >
                            {status === "all" ? t("queue.all") : status === "in_progress" ? t("queue.inProgress") : status === "completed" ? t("queue.completed") : t("queue.waiting")}
                        </Button>
                    ))}

                    <div className="ml-auto flex w-full items-center gap-2 sm:w-auto">
                        <Search className="h-3.5 w-3.5 text-muted-foreground" />
                        <Input
                            value={searchTerm}
                            onChange={(event) => setSearchTerm(event.target.value)}
                            placeholder="Search by patient, department, or queue ID"
                            className="h-10 w-full sm:h-9 sm:w-72"
                            aria-label="Search queue entries"
                        />
                    </div>
                </div>
            </Card>

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                {loading ? (
                    <Card className="col-span-full flex h-40 items-center justify-center border-border/60 bg-card">
                        <Loader2 className="h-6 w-6 animate-spin text-primary" />
                    </Card>
                ) : queue.length === 0 ? (
                    <Card className="col-span-full flex h-44 flex-col items-center justify-center border-border/60 bg-card text-muted-foreground">
                        <ClipboardList className="mb-2 h-8 w-8 opacity-50" />
                        {t("queue.empty")}
                    </Card>
                ) : visibleQueue.length === 0 ? (
                    <Card className="col-span-full flex h-44 flex-col items-center justify-center border-border/60 bg-card text-muted-foreground">
                        <Search className="mb-2 h-7 w-7 opacity-60" />
                        No queue entries match this search.
                    </Card>
                ) : (
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    visibleQueue.map((item: any, index: number) => {
                        const queueId = Number(item.queue_id);
                        const routeDecision = routesByQueueId[queueId];
                        const routeError = routeErrorByQueueId[queueId];
                        const routeMessage = routeMessageByQueueId[queueId];

                        const selectedCandidate =
                            routeDecision &&
                                routeDecision.explanation_fields &&
                                typeof routeDecision.explanation_fields === "object" &&
                                "selected_candidate" in routeDecision.explanation_fields
                                ? (routeDecision.explanation_fields.selected_candidate as Record<string, unknown>)
                                : null;

                        const selectedFlags = selectedCandidate && Array.isArray(selectedCandidate.policy_flags)
                            ? selectedCandidate.policy_flags.map((flag) => String(flag))
                            : [];

                        return (
                            <Card key={item.queue_id} className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm">
                                <div className="mb-3 flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-muted-foreground">#{index + 1} • Queue ID {item.queue_id}</p>
                                        <p className="font-mono text-sm font-semibold">{item.patient_id || `P-${item.prediction_id}`}</p>
                                        {item.patient_name ? (
                                            <p className="text-xs text-muted-foreground">{item.patient_name}</p>
                                        ) : null}
                                    </div>
                                    <StatusBadge status={item.status} />
                                </div>

                                <div className="mb-3 flex flex-wrap items-center gap-2">
                                    <PriorityBadge category={item.priority_score >= 9 ? "Critical" : item.priority_score >= 7 ? "High" : item.priority_score >= 5 ? "Standard" : "Low"} />
                                    <span className="text-xs text-muted-foreground">Score: {Number(item.priority_score).toFixed(1)}</span>
                                    <span className="text-xs text-muted-foreground">ESI: {item.esi_level ?? "-"}</span>
                                    <span className="text-xs text-muted-foreground">Dept: {item.department}</span>
                                </div>

                                <div className="rounded-lg border border-border/50 bg-muted/20 p-2.5 text-xs text-muted-foreground">
                                    <p>Wait: {item.dynamic_estimated_wait || item.estimated_wait_time}</p>
                                    {item.monitoring_note ? (
                                        <p className={`mt-1 ${item.deteriorating ? "text-red-400" : "text-muted-foreground"}`}>{item.monitoring_note}</p>
                                    ) : null}
                                    {item.vitals_trend?.signals?.length ? (
                                        <p className="mt-1 text-amber-300">Trend: {item.vitals_trend.signals.join(", ")}</p>
                                    ) : null}
                                </div>

                                <div className="mt-3 rounded-lg border border-primary/20 bg-primary/5 p-3">
                                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                                        <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-primary">
                                            <Route className="h-3.5 w-3.5" />
                                            Route Recommendation
                                        </div>
                                        <div className="flex w-full flex-col gap-1 sm:w-auto sm:flex-row">
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                disabled={routeLoadingQueueId === queueId}
                                                onClick={() => handleRecommendRoute(item, index)}
                                                className="min-h-10 w-full px-2.5 sm:min-h-9 sm:w-auto"
                                            >
                                                {routeLoadingQueueId === queueId ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Route className="mr-1 h-3 w-3" />}
                                                Recommend
                                            </Button>
                                            <Button
                                                size="sm"
                                                disabled={routeAdmitQueueId === queueId}
                                                onClick={() => handleRouteAndAdmit(item, index)}
                                                className="min-h-10 w-full px-2.5 sm:min-h-9 sm:w-auto"
                                            >
                                                {routeAdmitQueueId === queueId ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Building2 className="mr-1 h-3 w-3" />}
                                                Route and Admit
                                            </Button>
                                        </div>
                                    </div>

                                    {routeError ? <p className="mb-2 text-xs text-destructive" role="alert">{routeError}</p> : null}
                                    {routeMessage ? <p className="mb-2 text-xs text-emerald-400" role="status" aria-live="polite">{routeMessage}</p> : null}

                                    {routeDecision ? (
                                        <div className="space-y-2 text-xs">
                                            <div className="rounded-md border border-border/50 bg-muted/30 p-2">
                                                <p className="font-semibold text-foreground">
                                                    {routeDecision.recommended_hospital_name} • {routeDecision.recommended_ward_name}
                                                </p>
                                                <p className="mt-1 text-muted-foreground">
                                                    Wait {routeDecision.estimated_wait_minutes} min ({routeDecision.estimated_wait_band}) • Overflow {routeDecision.overflow_risk}
                                                </p>
                                                {typeof routeDecision.inflow_rank === "number" ? (
                                                    <p className="text-muted-foreground">Inflow rank: {routeDecision.inflow_rank}</p>
                                                ) : null}
                                                <p className="text-muted-foreground">
                                                    Bed: {routeDecision.recommended_bed_label || routeDecision.recommended_bed_id || "No immediate bed"}
                                                </p>
                                                <p className="mt-1 text-muted-foreground">{routeDecision.route_reason}</p>
                                                {selectedFlags.length > 0 ? (
                                                    <p className="mt-1 text-blue-300">Policy: {selectedFlags.join(", ")}</p>
                                                ) : null}
                                            </div>

                                            {routeDecision.alternatives.length > 0 ? (
                                                <div>
                                                    <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Alternatives</p>
                                                    <div className="space-y-1">
                                                        {routeDecision.alternatives.slice(0, 3).map((alt, altIndex) => (
                                                            <div key={`${alt.ward_id}-${altIndex}`} className="rounded-md bg-black/10 px-2 py-1 text-[11px] text-muted-foreground">
                                                                {alt.hospital_name} • {alt.ward_name} • beds {alt.available_beds} • score {alt.composite_score.toFixed(3)}
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            ) : null}
                                        </div>
                                    ) : (
                                        <p className="text-xs text-muted-foreground">Generate a route to see recommendation reasoning and alternatives.</p>
                                    )}
                                </div>

                                <div className="mt-3 flex flex-col justify-end gap-1 sm:flex-row">
                                    {item.status === "waiting" && (
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            disabled={actioning === item.queue_id}
                                            onClick={() => handleStatusChange(item.queue_id, "in_progress")}
                                            className="min-h-10 w-full px-3 sm:min-h-9 sm:w-auto"
                                        >
                                            {actioning === item.queue_id ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Play className="mr-1 h-3 w-3" />}
                                            Start
                                        </Button>
                                    )}
                                    {item.status === "in_progress" && (
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            disabled={actioning === item.queue_id}
                                            onClick={() => handleStatusChange(item.queue_id, "completed")}
                                            className="min-h-10 w-full px-3 sm:min-h-9 sm:w-auto"
                                        >
                                            {actioning === item.queue_id ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <CheckCircle className="mr-1 h-3 w-3" />}
                                            Complete
                                        </Button>
                                    )}
                                </div>
                            </Card>
                        );
                    })
                )}
            </div>
        </div>
    );
}
