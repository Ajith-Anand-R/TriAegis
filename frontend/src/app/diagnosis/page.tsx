"use client";

import { useEffect, useMemo, useState } from "react";
import { getConstants, predictSingle, predictAndSave } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RiskBadge, PriorityBadge } from "@/components/risk-badge";
import {
  Activity,
  ClipboardCheck,
  FlaskConical,
  Gauge,
  HeartPulse,
  Layers3,
  Loader2,
  RotateCcw,
  ShieldAlert,
  Stethoscope,
  Thermometer,
  UserRound,
} from "lucide-react";

type DiagnosisResult = {
  prediction?: {
    risk_level?: string;
    priority_score?: number;
    priority_category?: string;
    confidence?: number;
    esi_level?: number;
    department?: string;
    department_reason?: string;
    estimated_wait_time?: string;
    manual_review_required?: boolean;
    differential_diagnosis?: Array<{
      diagnosis: string;
      likelihood: string;
      rationale: string;
    }>;
  };
  manual_review_recommended?: boolean;
  clinical_explanation?: string;
};

export default function DiagnosisPage() {
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [conditions, setConditions] = useState<string[]>([]);

  const [patientId, setPatientId] = useState("DX-001");
  const [patientName, setPatientName] = useState("");
  const [age, setAge] = useState(45);
  const [gender, setGender] = useState("Male");
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [systolic, setSystolic] = useState(120);
  const [diastolic, setDiastolic] = useState(80);
  const [heartRate, setHeartRate] = useState(72);
  const [temperature, setTemperature] = useState(98.6);
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    getConstants()
      .then((data) => {
        setSymptoms(data.symptoms || []);
        setConditions(data.conditions || []);
      })
      .catch(() => {
        setSymptoms([]);
        setConditions([]);
      });
  }, []);

  const payload = useMemo(
    () => ({
      Patient_ID: patientId,
      "Patient Name": patientName.trim(),
      Age: age,
      Gender: gender,
      Symptoms: selectedSymptoms.join(","),
      "Blood Pressure": `${systolic}/${diastolic}`,
      "Heart Rate": heartRate,
      Temperature: temperature,
      "Pre-Existing Conditions": selectedConditions.length ? selectedConditions.join(",") : "none",
    }),
    [patientId, patientName, age, gender, selectedSymptoms, systolic, diastolic, heartRate, temperature, selectedConditions]
  );

  const toggleItem = (values: string[], item: string, setter: (next: string[]) => void) => {
    setter(values.includes(item) ? values.filter((entry) => entry !== item) : [...values, item]);
  };

  const resetForm = () => {
    setPatientId("DX-001");
    setPatientName("");
    setAge(45);
    setGender("Male");
    setSelectedSymptoms([]);
    setSystolic(120);
    setDiastolic(80);
    setHeartRate(72);
    setTemperature(98.6);
    setSelectedConditions([]);
    setResult(null);
    setErrorMessage(null);
  };

  const validateBeforeRun = (): string | null => {
    if (selectedSymptoms.length === 0) {
      return "Select at least one symptom before running diagnosis.";
    }
    return null;
  };

  const runDiagnosis = async () => {
    const validationError = validateBeforeRun();
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }

    setErrorMessage(null);
    setLoading(true);
    try {
      const data = (await predictSingle(payload)) as DiagnosisResult;
      setResult(data);
    } catch (error: unknown) {
      setErrorMessage(error instanceof Error ? error.message : "Diagnosis failed");
    } finally {
      setLoading(false);
    }
  };

  const runDiagnosisAndSave = async () => {
    const validationError = validateBeforeRun();
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }

    setErrorMessage(null);
    setSaving(true);
    try {
      const data = (await predictAndSave(payload)) as DiagnosisResult;
      setResult(data);
    } catch (error: unknown) {
      setErrorMessage(error instanceof Error ? error.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const prediction = result?.prediction;

  const recommendations = useMemo(() => {
    if (!prediction) return [] as string[];
    const items: string[] = [];

    if (result?.manual_review_recommended || prediction.manual_review_required) {
      items.push("Manual clinician review required before final disposition.");
    }

    if ((prediction.risk_level || "") === "High") {
      items.push("Initiate urgent care pathway and prioritize immediate physician evaluation.");
    } else if ((prediction.risk_level || "") === "Medium") {
      items.push("Perform focused diagnostic workup and monitor vital signs closely.");
    } else {
      items.push("Proceed with standard triage workflow and routine reassessment.");
    }

    if ((prediction.esi_level || 5) <= 2) {
      items.push("Place patient in high-acuity observation area and reassess frequently.");
    }

    if (prediction.department) {
      items.push(`Route patient to ${prediction.department} (${prediction.department_reason || "specialist review"}).`);
    }

    return items;
  }, [prediction, result?.manual_review_recommended]);

  return (
    <div className="space-y-7" aria-busy={loading || saving}>
      <p role="status" aria-live="polite" className="sr-only">{errorMessage || ""}</p>
      <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-sky-500/15 via-card to-emerald-500/10 p-4 sm:p-6">
        <div className="pointer-events-none absolute -right-14 -top-14 h-40 w-40 rounded-full bg-sky-500/20 blur-2xl" />
        <div className="pointer-events-none absolute -bottom-16 -left-8 h-44 w-44 rounded-full bg-emerald-400/15 blur-2xl" />

        <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-sky-300">
              Clinical Decision Support
            </p>
            <h1 className="font-display mt-1 flex items-center gap-2 text-2xl font-semibold tracking-tight md:text-3xl">
              <Stethoscope className="h-7 w-7 text-primary" />
              Diagnosis and Recommendations
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
              Use structured triage inputs to generate differential diagnosis, care pathway suggestions, and manual review flags.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
            <span className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-2.5 py-1.5 text-sky-200">Differential</span>
            <span className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-1.5 text-indigo-200">Severity</span>
            <span className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1.5 text-emerald-200">Recommendations</span>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <div className="xl:col-span-5">
          <Card className="space-y-4 border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
            <div className="flex items-center gap-2">
              <Layers3 className="h-4 w-4 text-primary" />
              <h2 className="font-display text-base font-semibold">Patient Input</h2>
            </div>

            <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Demographics</p>

              <div className="mb-3">
                <Label className="mb-1.5 block text-xs text-muted-foreground">Patient ID</Label>
                <Input value={patientId} onChange={(event) => setPatientId(event.target.value)} className="bg-input/60" />
              </div>

              <div className="mb-3">
                <Label className="mb-1.5 block text-xs text-muted-foreground">Patient Name</Label>
                <Input
                  value={patientName}
                  onChange={(event) => setPatientName(event.target.value)}
                  className="bg-input/60"
                  placeholder="e.g. John Doe"
                />
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <Label className="mb-1.5 block text-xs text-muted-foreground">Age</Label>
                  <Input type="number" value={age} onChange={(event) => setAge(Number(event.target.value))} className="bg-input/60" />
                </div>
                <div>
                  <Label className="mb-1.5 block text-xs text-muted-foreground">Gender</Label>
                  <select
                    value={gender}
                    onChange={(event) => setGender(event.target.value)}
                    aria-label="Gender"
                    title="Gender"
                    className="flex h-10 w-full rounded-md border border-input bg-input/60 px-3 py-1 text-sm sm:h-9"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
              <p className="mb-3 flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                <UserRound className="h-3.5 w-3.5" />
                Symptoms ({selectedSymptoms.length})
              </p>
              <div className="max-h-36 overflow-y-auto rounded-lg border border-border/50 bg-input/30 p-2">
                <div className="flex flex-wrap gap-1.5">
                  {symptoms.map((item) => (
                    <button
                      key={item}
                      type="button"
                      onClick={() => toggleItem(selectedSymptoms, item, setSelectedSymptoms)}
                      aria-pressed={selectedSymptoms.includes(item)}
                      className={`rounded-md px-2 py-1 text-xs transition-all ${
                        selectedSymptoms.includes(item)
                          ? "border border-primary/30 bg-primary/20 font-medium text-primary"
                          : "border border-transparent bg-muted/50 text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground">
                {selectedSymptoms.length === 0
                  ? "Choose at least one symptom to enable diagnosis actions."
                  : "Symptom profile is ready for model evaluation."}
              </p>
            </div>

            <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
              <p className="mb-3 flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                <HeartPulse className="h-3.5 w-3.5" />
                Vitals
              </p>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <Label className="mb-1.5 block text-xs text-muted-foreground">BP Systolic</Label>
                  <Input type="number" value={systolic} onChange={(event) => setSystolic(Number(event.target.value))} className="bg-input/60" />
                </div>
                <div>
                  <Label className="mb-1.5 block text-xs text-muted-foreground">BP Diastolic</Label>
                  <Input type="number" value={diastolic} onChange={(event) => setDiastolic(Number(event.target.value))} className="bg-input/60" />
                </div>
              </div>

              <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <Label className="mb-1.5 flex items-center gap-1 text-xs text-muted-foreground">
                    <Gauge className="h-3 w-3" />
                    Heart Rate
                  </Label>
                  <Input type="number" value={heartRate} onChange={(event) => setHeartRate(Number(event.target.value))} className="bg-input/60" />
                </div>
                <div>
                  <Label className="mb-1.5 flex items-center gap-1 text-xs text-muted-foreground">
                    <Thermometer className="h-3 w-3" />
                    Temperature (F)
                  </Label>
                  <Input type="number" step={0.1} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} className="bg-input/60" />
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Pre-Existing Conditions ({selectedConditions.length})
              </p>
              <div className="flex flex-wrap gap-1.5">
                {conditions.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => toggleItem(selectedConditions, item, setSelectedConditions)}
                    aria-pressed={selectedConditions.includes(item)}
                    className={`rounded-md px-2 py-1 text-xs transition-all ${
                      selectedConditions.includes(item)
                        ? "border border-primary/30 bg-primary/20 font-medium text-primary"
                        : "border border-transparent bg-muted/50 text-muted-foreground hover:bg-muted"
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <Button
                onClick={runDiagnosis}
                disabled={loading || selectedSymptoms.length === 0}
                className="min-h-10 bg-gradient-to-r from-sky-600 to-indigo-600 px-3 text-white shadow-lg shadow-sky-500/20 hover:from-sky-500 hover:to-indigo-500 sm:min-h-9"
              >
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Activity className="mr-2 h-4 w-4" />}
                Run Diagnosis
              </Button>
              <Button onClick={runDiagnosisAndSave} disabled={saving || selectedSymptoms.length === 0} variant="outline" className="min-h-10 border-primary/30 px-3 sm:min-h-9">
                {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ClipboardCheck className="mr-2 h-4 w-4" />}
                Diagnose and Save
              </Button>
            </div>

            <Button type="button" variant="ghost" onClick={resetForm} className="min-h-10 w-full justify-center border border-border/60 px-3 sm:min-h-9">
              <RotateCcw className="mr-2 h-4 w-4" />
              Reset Form
            </Button>

            {errorMessage ? (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200" role="alert">
                {errorMessage}
              </div>
            ) : null}
          </Card>
        </div>

        <div className="xl:col-span-7">
          <Card className="space-y-4 border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
            <div className="flex items-center justify-between gap-3">
              <h2 className="font-display flex items-center gap-2 text-lg font-semibold">
                <FlaskConical className="h-5 w-5 text-primary" />
                Diagnosis Output
              </h2>
              {prediction ? (
                <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
                  Results Ready
                </span>
              ) : null}
            </div>

            {!prediction ? (
              <div className="flex h-72 flex-col items-center justify-center rounded-xl border border-dashed border-border/70 bg-muted/20 text-center text-sm text-muted-foreground">
                <Stethoscope className="mb-3 h-7 w-7 text-muted-foreground/70" />
                <p className="font-medium text-foreground">No diagnosis result yet</p>
                <p className="mt-1 max-w-sm text-xs text-muted-foreground">
                  Fill in patient details and run diagnosis to see differential findings and care recommendations.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <RiskBadge level={prediction.risk_level || "Low"} size="lg" />
                  <PriorityBadge category={prediction.priority_category || "Standard"} />
                  <span className="rounded-lg border border-border/60 bg-muted px-2.5 py-1 text-xs">ESI {prediction.esi_level || "-"}</span>
                  <span className="rounded-lg border border-border/60 bg-muted px-2.5 py-1 text-xs">Confidence {((prediction.confidence || 0) * 100).toFixed(1)}%</span>
                </div>

                <div className="rounded-xl border border-border/60 bg-muted/20 p-4 text-sm">
                  <p>
                    <span className="text-muted-foreground">Recommended Department: </span>
                    <span className="font-semibold">{prediction.department}</span>
                  </p>
                  <p className="mt-1 text-muted-foreground">{prediction.department_reason}</p>
                  <p className="mt-2">
                    <span className="text-muted-foreground">Estimated wait: </span>
                    {prediction.estimated_wait_time}
                  </p>
                </div>

                {Array.isArray(prediction.differential_diagnosis) && prediction.differential_diagnosis.length > 0 ? (
                  <div className="rounded-xl border border-primary/25 bg-primary/5 p-4">
                    <p className="text-sm font-semibold">Differential Diagnosis</p>
                    <div className="mt-2 space-y-2">
                      {prediction.differential_diagnosis.map((item, index) => (
                        <div key={index} className="rounded-lg border border-border/60 bg-card/80 p-3 text-sm">
                          <p className="font-semibold">{item.diagnosis}</p>
                          <p className="text-xs text-muted-foreground">Likelihood: {item.likelihood}</p>
                          <p className="mt-0.5 text-xs text-muted-foreground">{item.rationale}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 p-4">
                  <p className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
                    <ClipboardCheck className="h-4 w-4" />
                    Suggestions and Recommendations
                  </p>
                  <ul className="mt-2 list-disc space-y-1 pl-4 text-sm text-muted-foreground">
                    {recommendations.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>

                {result?.manual_review_recommended || prediction.manual_review_required ? (
                  <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
                    <p className="flex items-center gap-2 font-semibold">
                      <ShieldAlert className="h-4 w-4" />
                      Manual Review Alert
                    </p>
                    <p className="mt-1">Model output should be validated by a clinician before final decisions.</p>
                  </div>
                ) : null}

                {result?.clinical_explanation ? (
                  <div className="rounded-xl border border-blue-500/25 bg-blue-500/5 p-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-blue-300">Clinical Explanation</p>
                    <pre className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">{String(result.clinical_explanation)}</pre>
                  </div>
                ) : null}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
