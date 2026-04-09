"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { getConstants, getCurrentUser, getDashboard, predictSingle, predictAndSave, predictBatch, getSymptomDialogue, routeAndAdmit, routePatient } from "@/lib/api";
import type { RouteDecision } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { RiskBadge, PriorityBadge } from "@/components/risk-badge";
import { MetricCard } from "@/components/metric-card";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from "recharts";
import {
    Stethoscope,
    Save,
    Loader2,
    Sparkles,
    Heart,
    Thermometer,
    Gauge,
    User,
    Shield,
    Upload,
    FileSpreadsheet,
    Zap,
    Route,
    Building2,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";
import { useRouter } from "next/navigation";
import { useVoiceAssistant } from "@/components/voice-assistant-context";

const RISK_COLORS: Record<string, string> = {
    Low: "#22c55e",
    Medium: "#f59e0b",
    High: "#ef4444",
};

export default function AnalysisPage() {
    const { t } = useI18n();
    const router = useRouter();
    const voice = useVoiceAssistant();
    const [symptoms, setSymptoms] = useState<string[]>([]);
    const [conditions, setConditions] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [nextSerial, setNextSerial] = useState(1);

    // Form state
    const [patientId, setPatientId] = useState("P001");
    const [patientName, setPatientName] = useState("");
    const [age, setAge] = useState(40);
    const [gender, setGender] = useState("Male");
    const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
    const [systolic, setSystolic] = useState(120);
    const [diastolic, setDiastolic] = useState(80);
    const [heartRate, setHeartRate] = useState(70);
    const [temperature, setTemperature] = useState(98.6);
    const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
    const batchFileInputRef = useRef<HTMLInputElement | null>(null);

    const [batchFile, setBatchFile] = useState<File | null>(null);
    const [batchLoading, setBatchLoading] = useState(false);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [batchData, setBatchData] = useState<any>(null);
    const [batchError, setBatchError] = useState<string | null>(null);
    const [dialogueLoading, setDialogueLoading] = useState(false);
    const [dialogueData, setDialogueData] = useState<Record<string, unknown> | null>(null);

    // Result
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [result, setResult] = useState<any>(null);
    const [routeDecision, setRouteDecision] = useState<RouteDecision | null>(null);
    const [routeLoading, setRouteLoading] = useState(false);
    const [routeAdmitLoading, setRouteAdmitLoading] = useState(false);
    const [routeError, setRouteError] = useState<string | null>(null);
    const [routeMessage, setRouteMessage] = useState<string | null>(null);

    useEffect(() => {
        getConstants().then((data) => {
            setSymptoms(data.symptoms);
            setConditions(data.conditions);
        }).catch(() => { });
        getCurrentUser().then((user) => {
            if (user.role === "Admin") {
                router.replace("/queue");
            }
        }).catch(() => { });
        getDashboard().then((data) => {
            const next = (data.total_predictions || 0) + 1;
            setNextSerial(next);
            setPatientId(`P${String(next).padStart(3, "0")}`);
        }).catch(() => { });
    }, [router]);

    const buildPayload = useCallback(() => ({
        Patient_ID: patientId,
        "Patient Name": patientName.trim(),
        Age: age,
        Gender: gender,
        Symptoms: selectedSymptoms.join(","),
        "Blood Pressure": `${systolic}/${diastolic}`,
        "Heart Rate": heartRate,
        Temperature: temperature,
        "Pre-Existing Conditions": selectedConditions.length ? selectedConditions.join(",") : "none",
    }), [patientId, patientName, age, gender, selectedSymptoms, systolic, diastolic, heartRate, temperature, selectedConditions]);

    const handleAnalyze = async () => {
        setLoading(true);
        setRouteDecision(null);
        setRouteError(null);
        setRouteMessage(null);
        try {
            const data = await predictSingle(buildPayload());
            setResult(data);
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : t("analysis.failed");
            alert(msg);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        setRouteDecision(null);
        setRouteError(null);
        setRouteMessage(null);
        try {
            const data = await predictAndSave(buildPayload());
            setResult(data);
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : t("analysis.saveFailed");
            alert(msg);
        } finally {
            setSaving(false);
        }
    };

    const buildRoutePayload = () => {
        const prediction = result?.prediction;
        if (!prediction) {
            throw new Error("Run analysis first to generate routing context");
        }

        const priorityScore = Number(prediction.priority_score || 1);
        const riskLevel =
            priorityScore >= 8.5
                ? "High"
                : priorityScore >= 5
                    ? "Medium"
                    : String(prediction.risk_level || "Low");

        return {
            patient_id: String(result?.patient?.Patient_ID || patientId),
            risk_level: riskLevel,
            priority_score: priorityScore,
            department: String(prediction.department || "General Medicine"),
            queue_ahead: Math.max(0, Number(prediction.queue_position || 1) - 1),
        };
    };

    const handleRecommendRoute = async () => {
        setRouteLoading(true);
        setRouteError(null);
        setRouteMessage(null);

        try {
            const response = await routePatient(buildRoutePayload());
            const routing = response.routing as RouteDecision | undefined;
            if (!routing) {
                throw new Error("Routing response was empty");
            }
            setRouteDecision(routing);
            setRouteMessage(`Recommended ${routing.recommended_hospital_name} / ${routing.recommended_ward_name}`);
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Route recommendation failed";
            setRouteError(message);
        } finally {
            setRouteLoading(false);
        }
    };

    const handleRouteAndAdmit = async () => {
        setRouteAdmitLoading(true);
        setRouteError(null);
        setRouteMessage(null);

        try {
            const response = await routeAndAdmit(buildRoutePayload());
            const routing = response.routing as RouteDecision | undefined;
            if (routing) {
                setRouteDecision(routing);
            }

            if (response.admitted) {
                const admission = (response.admission || {}) as Record<string, unknown>;
                const bedLabel = String(admission.bed_label || admission.bed_id || "bed reserved");
                setRouteMessage(`Admitted via route workflow. Reserved ${bedLabel}.`);
            } else {
                setRouteError(String(response.admit_error || "Route generated but direct admission was not possible"));
            }
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : "Route-to-admit failed";
            setRouteError(message);
        } finally {
            setRouteAdmitLoading(false);
        }
    };

    const handleBatchUpload = async () => {
        if (!batchFile) return;
        const lowerName = batchFile.name.toLowerCase();
        if (!(lowerName.endsWith(".csv") || lowerName.endsWith(".xlsx") || lowerName.endsWith(".xls"))) {
            setBatchError(t("batch.fileTypeError"));
            return;
        }
        setBatchLoading(true);
        setBatchError(null);
        try {
            const data = await predictBatch(batchFile);
            setBatchData(data);
        } catch (e: unknown) {
            setBatchError(e instanceof Error ? e.message : t("batch.uploadFailed"));
        } finally {
            setBatchLoading(false);
        }
    };

    const handleSymptomDeepening = async () => {
        const complaint = selectedSymptoms[0] || "general symptoms";
        setDialogueLoading(true);
        setDialogueData(null);
        try {
            const response = await getSymptomDialogue({
                presenting_complaint: complaint,
                transcript: [],
                known_patient: {
                    Age: age,
                    Gender: gender,
                    Symptoms: selectedSymptoms.join(","),
                    "Pre-Existing Conditions": selectedConditions.join(","),
                },
                model: "phi3:mini",
                max_followup_questions: 3,
            });
            setDialogueData(response);
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : "Symptom deepening failed";
            setDialogueData({
                source: "error",
                error: msg,
                follow_up_questions: [],
                structured: {
                    extracted: { chief_complaint: complaint },
                    red_flags: [],
                    urgency_hint: "requires_clinical_assessment",
                },
            });
        } finally {
            setDialogueLoading(false);
        }
    };

    const toggleItem = (list: string[], item: string, setter: (v: string[]) => void) => {
        setter(list.includes(item) ? list.filter((i) => i !== item) : [...list, item]);
    };

    // Keep a mutable ref to the latest result so the voice bot can read it
    const resultRef = useRef(result);
    useEffect(() => { resultRef.current = result; }, [result]);

    // Keep mutable refs for symptom/condition options
    const symptomsRef = useRef(symptoms);
    useEffect(() => { symptomsRef.current = symptoms; }, [symptoms]);
    const conditionsRef = useRef(conditions);
    useEffect(() => { conditionsRef.current = conditions; }, [conditions]);

    // Register form handlers with the global voice assistant
    useEffect(() => {
        voice.registerFormHandlers({
            setAge,
            setGender,
            setSelectedSymptoms,
            setSelectedConditions,
            triggerFileUpload: () => batchFileInputRef.current?.click(),
            triggerAnalyze: async () => {
                setLoading(true);
                try {
                    const data = await predictSingle(buildPayload());
                    setResult(data);
                } catch (e: unknown) {
                    const msg = e instanceof Error ? e.message : "Analysis failed";
                    throw new Error(msg);
                } finally {
                    setLoading(false);
                }
            },
            getSymptomOptions: () => symptomsRef.current,
            getConditionOptions: () => conditionsRef.current,
            getResult: () => resultRef.current,
        });

        return () => {
            voice.unregisterFormHandlers();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [buildPayload]);

    const pred = result?.prediction;
    const expl = result?.explanation;
    const displayRiskLevel = pred
        ? (pred.priority_score >= 9
            ? "High"
            : pred.priority_score >= 7
                ? (pred.risk_level === "Low" ? "Medium" : pred.risk_level)
                : pred.risk_level)
        : undefined;

    const probData = pred
        ? [
            { name: "Low", value: pred.probabilities.Low },
            { name: "Medium", value: pred.probabilities.Medium },
            { name: "High", value: pred.probabilities.High },
        ]
        : [];

    const shapData = expl?.explanation?.top_contributors?.map(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (c: any) => ({
            feature: c.feature,
            impact: c.impact,
        })
    ) || [];

    const selectedRouteCandidate =
        routeDecision &&
            routeDecision.explanation_fields &&
            typeof routeDecision.explanation_fields === "object" &&
            "selected_candidate" in routeDecision.explanation_fields
            ? (routeDecision.explanation_fields.selected_candidate as Record<string, unknown>)
            : null;

    const routePolicyFlags = selectedRouteCandidate && Array.isArray(selectedRouteCandidate.policy_flags)
        ? selectedRouteCandidate.policy_flags.map((flag) => String(flag))
        : [];
    const liveStatusMessage = routeError || routeMessage || batchError || "";

    return (
        <div className="space-y-7" aria-busy={loading || saving || batchLoading}>
            <p role="status" aria-live="polite" className="sr-only">{liveStatusMessage}</p>
            <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-blue-500/15 via-card to-cyan-500/10 p-4 sm:p-6">
                <div className="pointer-events-none absolute -right-16 -top-12 h-44 w-44 rounded-full bg-blue-500/20 blur-2xl" />
                <div className="pointer-events-none absolute -bottom-16 -left-8 h-48 w-48 rounded-full bg-cyan-400/15 blur-3xl" />

                <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div>
                        <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-cyan-300">
                            Triage Intelligence Workspace
                        </p>
                        <h1 className="font-display mt-1 flex items-center gap-2 text-2xl font-semibold tracking-tight md:text-3xl">
                            <Stethoscope className="h-7 w-7 text-primary" />
                            {t("analysis.title")}
                        </h1>
                        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
                            {t("analysis.subtitle")}
                        </p>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
                        <span className="rounded-lg border border-blue-500/30 bg-blue-500/10 px-2.5 py-1.5 text-blue-200">Single Analysis</span>
                        <span className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-1.5 text-cyan-200">Route and Admit</span>
                        <span className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1.5 text-emerald-200">Explainability</span>
                    </div>
                </div>
            </section>

            <Card className="border-border/60 bg-gradient-to-r from-card to-card/90 p-4 shadow-sm">
                <div className="flex flex-wrap items-center gap-3">
                    <p className="font-display text-sm font-semibold">{t("batch.uploadTitle")}</p>
                    <label className="cursor-pointer rounded-lg border border-primary/30 bg-primary/10 px-3 py-2 text-xs font-medium text-primary transition-all hover:bg-primary/20">
                        <FileSpreadsheet className="mr-1 inline h-3.5 w-3.5" />
                        {t("batch.chooseFile")}
                        <input
                            type="file"
                            accept=".csv,.xlsx,.xls"
                            className="hidden"
                            ref={batchFileInputRef}
                            onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                        />
                    </label>
                    {batchFile ? <span className="text-xs text-muted-foreground">{batchFile.name}</span> : null}
                    <Button
                        size="sm"
                        onClick={handleBatchUpload}
                        disabled={!batchFile || batchLoading}
                        className="min-h-10 bg-gradient-to-r from-blue-600 to-cyan-600 px-3 text-white shadow-lg shadow-blue-500/20 hover:from-blue-500 hover:to-cyan-500 sm:min-h-9"
                    >
                        {batchLoading ? <><Upload className="mr-1 h-3.5 w-3.5 animate-pulse" /> {t("batch.processing")}</> : <><Zap className="mr-1 h-3.5 w-3.5" /> {t("batch.run")}</>}
                    </Button>
                </div>
                {batchError ? <p className="mt-2 text-sm text-destructive" role="alert">{batchError}</p> : null}
                {batchData?.risk_counts ? (
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                        <div className="rounded-md border border-border/60 bg-muted/30 p-2">Total: <span className="font-semibold">{batchData.risk_counts.total}</span></div>
                        <div className="rounded-md border border-red-500/20 bg-red-500/10 p-2">High: <span className="font-semibold">{batchData.risk_counts.high}</span></div>
                        <div className="rounded-md border border-amber-500/20 bg-amber-500/10 p-2">Medium: <span className="font-semibold">{batchData.risk_counts.medium}</span></div>
                        <div className="rounded-md border border-emerald-500/20 bg-emerald-500/10 p-2">Low: <span className="font-semibold">{batchData.risk_counts.low}</span></div>
                    </div>
                ) : null}
            </Card>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
                {/* ===== Column 1: Input ===== */}
                <div className="xl:col-span-4">
                    <Card className="h-full border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                        <h3 className="font-display mb-4 flex items-center gap-2 text-base font-semibold">
                            <User className="h-4 w-4 text-primary" /> {t("analysis.patientInput")}
                        </h3>

                        <div className="space-y-4">
                            {/* Patient ID */}
                            <div>
                                <Label className="mb-1.5 block text-xs text-muted-foreground">{t("analysis.patientId")}</Label>
                                <Input
                                    value={patientId}
                                    onChange={(e) => setPatientId(e.target.value)}
                                    className="bg-input/50 font-mono text-xs"
                                    readOnly
                                />
                            </div>

                            <div>
                                <Label className="mb-1.5 block text-xs text-muted-foreground">Patient Name</Label>
                                <Input
                                    value={patientName}
                                    onChange={(e) => setPatientName(e.target.value)}
                                    className="bg-input/50"
                                    placeholder="e.g. John Doe"
                                />
                            </div>

                            {/* Age & Gender */}
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                <div>
                                    <Label className="mb-1.5 block text-xs text-muted-foreground">{t("analysis.age")}</Label>
                                    <Input
                                        type="number"
                                        min={0}
                                        max={120}
                                        value={age}
                                        onChange={(e) => setAge(Number(e.target.value))}
                                        className="bg-input/50"
                                    />
                                </div>
                                <div>
                                    <Label htmlFor="analysis-gender" className="mb-1.5 block text-xs text-muted-foreground">{t("analysis.gender")}</Label>
                                    <select
                                        id="analysis-gender"
                                        aria-label={t("analysis.gender")}
                                        title={t("analysis.gender")}
                                        value={gender}
                                        onChange={(e) => setGender(e.target.value)}
                                        className="flex h-10 w-full rounded-md border border-input bg-input/50 px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring sm:h-9"
                                    >
                                        <option value="Male">{t("common.male")}</option>
                                        <option value="Female">{t("common.female")}</option>
                                        <option value="Other">{t("common.other")}</option>
                                    </select>
                                </div>
                            </div>

                            {/* Symptoms */}
                            <div>
                                <Label className="mb-1.5 block text-xs text-muted-foreground">
                                    {t("analysis.symptoms")} ({selectedSymptoms.length} {t("common.selected")})
                                </Label>
                                {/* Voice assistant status indicator (when active) */}
                                {voice.state.active && (
                                    <div className="mt-1 mb-1 flex items-center gap-2 rounded-md border border-primary/20 bg-primary/5 px-3 py-1.5">
                                        <span className="relative flex h-2.5 w-2.5">
                                            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
                                            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-primary" />
                                        </span>
                                        <span className="text-[11px] font-medium text-primary">Voice Assistant Active — use the floating panel to interact</span>
                                    </div>
                                )}
                                <div className="mt-1 max-h-36 overflow-y-auto rounded-lg border border-border/50 bg-input/30 p-2">
                                    <div className="flex flex-wrap gap-1.5">
                                        {symptoms.map((s) => (
                                            <button
                                                key={s}
                                                type="button"
                                                onClick={() => toggleItem(selectedSymptoms, s, setSelectedSymptoms)}
                                                aria-pressed={selectedSymptoms.includes(s)}
                                                className={`rounded-md px-2 py-1 text-xs transition-all ${selectedSymptoms.includes(s)
                                                    ? "bg-primary/20 text-primary border border-primary/30 font-medium"
                                                    : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
                                                    }`}
                                            >
                                                {s}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Vitals */}
                            <Separator />
                            <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                <Heart className="h-3.5 w-3.5" /> {t("analysis.vitals")}
                            </h4>
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                <div>
                                    <Label className="mb-1.5 block text-xs text-muted-foreground">
                                        <Gauge className="inline h-3 w-3 mr-1" />{t("analysis.bpSystolic")}
                                    </Label>
                                    <Input type="number" min={70} max={260} value={systolic} onChange={(e) => setSystolic(Number(e.target.value))} className="bg-input/50" />
                                </div>
                                <div>
                                    <Label className="mb-1.5 block text-xs text-muted-foreground">{t("analysis.bpDiastolic")}</Label>
                                    <Input type="number" min={40} max={160} value={diastolic} onChange={(e) => setDiastolic(Number(e.target.value))} className="bg-input/50" />
                                </div>
                            </div>
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                <div>
                                    <Label className="mb-1.5 block text-xs text-muted-foreground">
                                        <Heart className="inline h-3 w-3 mr-1" />{t("analysis.heartRate")}
                                    </Label>
                                    <Input type="number" min={30} max={220} value={heartRate} onChange={(e) => setHeartRate(Number(e.target.value))} className="bg-input/50" />
                                </div>
                                <div>
                                    <Label className="mb-1.5 block text-xs text-muted-foreground">
                                        <Thermometer className="inline h-3 w-3 mr-1" />{t("analysis.tempF")}
                                    </Label>
                                    <Input type="number" min={90} max={112} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="bg-input/50" />
                                </div>
                            </div>

                            {/* Conditions */}
                            <Separator />
                            <div>
                                <Label className="mb-1.5 block text-xs text-muted-foreground">
                                    <Shield className="inline h-3 w-3 mr-1" />
                                    {t("analysis.conditions")} ({selectedConditions.length})
                                </Label>
                                <div className="mt-1 flex flex-wrap gap-1.5">
                                    {conditions.map((c) => (
                                        <button
                                            key={c}
                                            type="button"
                                            onClick={() => toggleItem(selectedConditions, c, setSelectedConditions)}
                                            aria-pressed={selectedConditions.includes(c)}
                                            className={`rounded-md px-2 py-1 text-xs transition-all ${selectedConditions.includes(c)
                                                ? "bg-primary/20 text-primary border border-primary/30 font-medium"
                                                : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
                                                }`}
                                        >
                                            {c}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Analyze Button */}
                            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                                <Button
                                    type="button"
                                    onClick={handleSymptomDeepening}
                                    disabled={dialogueLoading || selectedSymptoms.length === 0}
                                    variant="outline"
                                    size="lg"
                                    className="border-primary/30"
                                >
                                    {dialogueLoading ? (
                                        <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Deepening...</>
                                    ) : (
                                        <><Sparkles className="mr-2 h-4 w-4" /> Symptom Deepening</>
                                    )}
                                </Button>
                                <Button
                                onClick={handleAnalyze}
                                disabled={loading}
                                className="min-h-10 w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold shadow-lg shadow-blue-500/20 transition-all hover:from-blue-500 hover:to-indigo-500 sm:min-h-9"
                                size="lg"
                            >
                                {loading ? (
                                    <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> {t("analysis.analyzing")}</>
                                ) : (
                                    <><Sparkles className="mr-2 h-4 w-4" /> {t("analysis.analyze")}</>
                                )}
                                </Button>
                            </div>

                            {dialogueData ? (
                                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-xs">
                                    {"error" in dialogueData && dialogueData.error ? (
                                        <div className="text-amber-300">
                                            <p className="font-semibold">Model Error</p>
                                            <p className="mt-1 text-muted-foreground">{String(dialogueData.error)}</p>
                                        </div>
                                    ) : (
                                        <>
                                            <p className="font-semibold text-primary">Follow-up Questions</p>
                                            <ul className="mt-2 list-disc space-y-1 pl-4 text-muted-foreground">
                                                {Array.isArray(dialogueData.follow_up_questions) && dialogueData.follow_up_questions.length > 0 ? (
                                                    dialogueData.follow_up_questions.map((q, idx) => (
                                                        <li key={idx}>{String(q)}</li>
                                                    ))
                                                ) : (
                                                    <li>No additional follow-up questions generated.</li>
                                                )}
                                            </ul>
                                            <p className="mt-2 text-[11px] text-muted-foreground">
                                                Source: {String(dialogueData.source || "unknown")}
                                            </p>
                                        </>
                                    )}
                                </div>
                            ) : null}
                        </div>
                    </Card>
                </div>

                {/* ===== Column 2: Results ===== */}
                <div className="xl:col-span-4">
                    <Card className="h-full border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                        <h3 className="font-display mb-4 text-base font-semibold">{t("analysis.resultsTitle")}</h3>

                        {pred ? (
                            <div className="space-y-5">
                                <RiskBadge level={displayRiskLevel || pred.risk_level} size="lg" />

                                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                    <MetricCard title={t("analysis.priorityScore")} value={`${pred.priority_score}/10`} variant={pred.priority_score >= 8 ? "danger" : pred.priority_score >= 5 ? "warning" : "success"} />
                                    <MetricCard title={t("analysis.confidence")} value={`${(pred.confidence * 100).toFixed(1)}%`} variant="info" />
                                </div>

                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-muted-foreground">{t("common.priority")}:</span>
                                    <PriorityBadge category={pred.priority_category} />
                                </div>

                                <div className="space-y-2 rounded-lg bg-muted/30 p-3">
                                    <div className="flex justify-between text-sm">
                                        <span className="text-muted-foreground">{t("analysis.queuePosition")}</span>
                                        <span className="font-semibold">{pred.queue_position}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                        <span className="text-muted-foreground">ESI Level</span>
                                        <span className="font-semibold">{pred.esi_level ?? "-"}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                        <span className="text-muted-foreground">{t("analysis.estWait")}</span>
                                        <span className="font-semibold">{pred.estimated_wait_time}</span>
                                    </div>
                                </div>

                                <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-xs">
                                    <p className="font-medium text-amber-300">Distribution Safety</p>
                                    <p className="mt-1 text-muted-foreground">
                                        OOD: {pred.out_of_distribution ? "Yes" : "No"} | Score: {typeof pred.ood_score === "number" ? pred.ood_score.toFixed(2) : "-"} / {typeof pred.ood_threshold === "number" ? pred.ood_threshold.toFixed(2) : "-"}
                                    </p>
                                    {pred.manual_review_required ? (
                                        <p className="mt-1 font-semibold text-amber-300">Manual review required before final disposition.</p>
                                    ) : null}
                                </div>

                                {Array.isArray(pred.differential_diagnosis) && pred.differential_diagnosis.length > 0 ? (
                                    <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-3 text-xs">
                                        <p className="font-medium text-violet-300">Differential Diagnosis</p>
                                        <div className="mt-2 space-y-2">
                                            {pred.differential_diagnosis.map((item: Record<string, unknown>, idx: number) => (
                                                <div key={idx} className="rounded-md bg-black/10 p-2">
                                                    <p className="font-semibold text-foreground">{String(item.diagnosis || "Unknown")}</p>
                                                    <p className="text-muted-foreground">Likelihood: {String(item.likelihood || "Unknown")}</p>
                                                    <p className="text-muted-foreground">{String(item.rationale || "")}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : null}

                                {result?.clinical_explanation ? (
                                    <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3">
                                        <p className="text-xs font-medium text-blue-400">{t("analysis.clinicalExplanation")}</p>
                                        <pre className="mt-1 whitespace-pre-wrap text-xs text-muted-foreground">{String(result.clinical_explanation)}</pre>
                                        {result?.manual_review_recommended ? (
                                            <p className="mt-2 text-xs font-semibold text-amber-400">
                                                ⚠️ {t("analysis.manualReview")} ({String(result?.confidence_band || "Low")} {t("analysis.confidence")})
                                            </p>
                                        ) : (
                                            <p className="mt-2 text-xs font-semibold text-emerald-400">
                                                ✅ {t("analysis.assistedTriage")} ({String(result?.confidence_band || "High")} {t("analysis.confidence")})
                                            </p>
                                        )}
                                    </div>
                                ) : null}

                                {/* Probabilities Chart */}
                                <div>
                                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">{t("analysis.riskProbabilities")}</p>
                                    <ResponsiveContainer width="100%" height={100}>
                                        <BarChart data={probData} layout="vertical" margin={{ left: 0, right: 0 }}>
                                            <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: "#888" }} />
                                            <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#ccc" }} width={55} />
                                            <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} />
                                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                                {probData.map((entry) => (
                                                    <Cell key={entry.name} fill={RISK_COLORS[entry.name]} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Department */}
                                <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                                    <p className="text-xs font-medium text-emerald-400">{t("analysis.recommendedDepartment")}</p>
                                    <p className="mt-1 text-sm font-semibold">{pred.department}</p>
                                    <p className="mt-0.5 text-xs text-muted-foreground">{pred.department_reason}</p>
                                </div>

                                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
                                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                                        <p className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-primary">
                                            <Route className="h-3.5 w-3.5" />
                                            Route Recommendation
                                        </p>
                                        <div className="flex w-full flex-col gap-1 sm:w-auto sm:flex-row">
                                            <Button
                                                type="button"
                                                size="sm"
                                                variant="outline"
                                                onClick={handleRecommendRoute}
                                                disabled={routeLoading || !pred}
                                                className="min-h-10 w-full px-2.5 sm:min-h-9 sm:w-auto"
                                            >
                                                {routeLoading ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Route className="mr-1 h-3 w-3" />}
                                                Recommend
                                            </Button>
                                            <Button
                                                type="button"
                                                size="sm"
                                                onClick={handleRouteAndAdmit}
                                                disabled={routeAdmitLoading || !pred}
                                                className="min-h-10 w-full px-2.5 sm:min-h-9 sm:w-auto"
                                            >
                                                {routeAdmitLoading ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Building2 className="mr-1 h-3 w-3" />}
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
                                                <p className="text-muted-foreground">
                                                    Bed: {routeDecision.recommended_bed_label || routeDecision.recommended_bed_id || "No immediate bed"}
                                                </p>
                                                <p className="mt-1 text-muted-foreground">{routeDecision.route_reason}</p>
                                                {routePolicyFlags.length > 0 ? (
                                                    <p className="mt-1 text-blue-300">Policy: {routePolicyFlags.join(", ")}</p>
                                                ) : null}
                                            </div>

                                            {routeDecision.alternatives.length > 0 ? (
                                                <div>
                                                    <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Alternatives</p>
                                                    <div className="space-y-1">
                                                        {routeDecision.alternatives.slice(0, 3).map((alt, idx) => (
                                                            <div key={`${alt.ward_id}-${idx}`} className="rounded-md bg-black/10 px-2 py-1 text-[11px] text-muted-foreground">
                                                                {alt.hospital_name} • {alt.ward_name} • beds {alt.available_beds} • score {alt.composite_score.toFixed(3)}
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            ) : null}
                                        </div>
                                    ) : (
                                        <p className="text-xs text-muted-foreground">Generate a route to inspect reasoning and alternatives before committing admission.</p>
                                    )}
                                </div>

                                {/* Actions */}
                                <div className="flex flex-col gap-2 sm:flex-row">
                                    <Button onClick={handleSave} disabled={saving} variant="outline" className="min-h-10 flex-1 px-3 sm:min-h-9" size="sm">
                                        {saving ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Save className="mr-1 h-3 w-3" />}
                                        {t("analysis.saveToDb")}
                                    </Button>
                                    <Button variant="outline" size="sm" className="min-h-10 px-3 sm:min-h-9" onClick={() => {
                                        setResult(null);
                                        setRouteDecision(null);
                                        setRouteError(null);
                                        setRouteMessage(null);
                                        const n = nextSerial + 1;
                                        setNextSerial(n);
                                        setPatientId(`P${String(n).padStart(3, "0")}`);
                                    }}>
                                        {t("common.clear")}
                                    </Button>
                                </div>
                            </div>
                        ) : (
                            <div className="flex h-64 items-center justify-center text-center text-sm text-muted-foreground">
                                {t("analysis.runToSeeResults")}
                            </div>
                        )}
                    </Card>
                </div>

                {/* ===== Column 3: Explanation ===== */}
                <div className="xl:col-span-4">
                    <Card className="h-full border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                        <h3 className="font-display mb-4 text-base font-semibold">{t("analysis.whyPrediction")}</h3>

                        {expl ? (
                            <div className="space-y-4">
                                {/* SHAP Chart */}
                                {shapData.length > 0 && (
                                    <div>
                                        <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">{t("analysis.topShapContributors")}</p>
                                        <ResponsiveContainer width="100%" height={shapData.length * 32 + 20}>
                                            <BarChart data={shapData} layout="vertical" margin={{ left: 10, right: 10 }}>
                                                <XAxis type="number" tick={{ fontSize: 10, fill: "#888" }} />
                                                <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: "#ccc" }} width={130} />
                                                <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, fontSize: 12 }} />
                                                <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                                                    {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                                    {shapData.map((entry: any, i: number) => (
                                                        <Cell key={i} fill={entry.impact >= 0 ? "#ef4444" : "#22c55e"} />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}

                                {/* Contributor Table */}
                                <div className="space-y-2 md:hidden">
                                    {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                    {expl.explanation.top_contributors.map((c: any, i: number) => (
                                        <article key={`contrib-mobile-${i}`} className="rounded-lg border border-border/60 bg-card/70 p-3">
                                            <div className="flex items-center justify-between gap-2">
                                                <p className="text-xs font-semibold">{c.feature}</p>
                                                <p className={`font-mono text-xs font-semibold ${c.impact >= 0 ? "text-red-400" : "text-emerald-400"}`}>
                                                    {c.impact >= 0 ? "+" : ""}{c.impact.toFixed(4)}
                                                </p>
                                            </div>
                                            <p className="mt-1 text-[11px] text-muted-foreground">{t("analysis.value")}: {String(c.value)}</p>
                                        </article>
                                    ))}
                                </div>

                                <div className="hidden md:block">
                                    <div className="mobile-scroll overflow-x-auto rounded-lg border border-border/50">
                                        <table className="w-full text-xs">
                                            <caption className="sr-only">Model contributor feature impact table</caption>
                                            <thead>
                                                <tr className="border-b border-border/50 bg-muted/30">
                                                    <th className="px-3 py-2 text-left font-medium text-muted-foreground">{t("analysis.feature")}</th>
                                                    <th className="px-3 py-2 text-left font-medium text-muted-foreground">{t("analysis.value")}</th>
                                                    <th className="px-3 py-2 text-right font-medium text-muted-foreground">{t("analysis.impact")}</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                                {expl.explanation.top_contributors.map((c: any, i: number) => (
                                                    <tr key={i} className="border-b border-border/30 last:border-0">
                                                        <td className="px-3 py-2 font-medium">{c.feature}</td>
                                                        <td className="px-3 py-2 text-muted-foreground">{String(c.value)}</td>
                                                        <td className={`px-3 py-2 text-right font-mono font-semibold ${c.impact >= 0 ? "text-red-400" : "text-emerald-400"}`}>
                                                            {c.impact >= 0 ? "+" : ""}{c.impact.toFixed(4)}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {/* Confidence Factors */}
                                <div className="space-y-1.5">
                                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{t("analysis.confidenceFactors")}</p>
                                    {expl.explanation.confidence_factors.map((f: string, i: number) => (
                                        <p key={i} className="text-xs text-muted-foreground">• {f}</p>
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <div className="flex h-64 items-center justify-center text-center text-sm text-muted-foreground">
                                {t("analysis.explainabilityAfter")}
                            </div>
                        )}
                    </Card>
                </div>
            </div>
        </div>
    );
}
