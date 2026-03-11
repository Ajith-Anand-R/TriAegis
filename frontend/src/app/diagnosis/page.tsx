"use client";

import { useEffect, useMemo, useState } from "react";
import { getConstants, predictSingle, predictAndSave } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RiskBadge, PriorityBadge } from "@/components/risk-badge";
import { Loader2, Stethoscope, ShieldAlert, ClipboardCheck } from "lucide-react";

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
      Age: age,
      Gender: gender,
      Symptoms: selectedSymptoms.join(","),
      "Blood Pressure": `${systolic}/${diastolic}`,
      "Heart Rate": heartRate,
      Temperature: temperature,
      "Pre-Existing Conditions": selectedConditions.length ? selectedConditions.join(",") : "none",
    }),
    [patientId, age, gender, selectedSymptoms, systolic, diastolic, heartRate, temperature, selectedConditions]
  );

  const toggleItem = (values: string[], item: string, setter: (next: string[]) => void) => {
    setter(values.includes(item) ? values.filter((entry) => entry !== item) : [...values, item]);
  };

  const runDiagnosis = async () => {
    setLoading(true);
    try {
      const data = (await predictSingle(payload)) as DiagnosisResult;
      setResult(data);
    } catch (error: unknown) {
      alert(error instanceof Error ? error.message : "Diagnosis failed");
    } finally {
      setLoading(false);
    }
  };

  const runDiagnosisAndSave = async () => {
    setSaving(true);
    try {
      const data = (await predictAndSave(payload)) as DiagnosisResult;
      setResult(data);
    } catch (error: unknown) {
      alert(error instanceof Error ? error.message : "Save failed");
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
    <div className="space-y-6">
      <div>
        <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
          <Stethoscope className="h-6 w-6 text-primary" />
          Clinical Diagnosis & Recommendation
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Enter patient details to generate differential diagnosis and triage recommendations.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <div className="xl:col-span-5">
          <Card className="border-border/50 bg-card p-6 space-y-4">
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5">Patient ID</Label>
              <Input value={patientId} onChange={(event) => setPatientId(event.target.value)} className="bg-input/50" />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5">Age</Label>
                <Input type="number" value={age} onChange={(event) => setAge(Number(event.target.value))} className="bg-input/50" />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5">Gender</Label>
                <select
                  value={gender}
                  onChange={(event) => setGender(event.target.value)}
                  aria-label="Gender"
                  title="Gender"
                  className="flex h-9 w-full rounded-md border border-input bg-input/50 px-3 py-1 text-sm"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            <div>
              <Label className="text-xs text-muted-foreground mb-1.5">Symptoms ({selectedSymptoms.length})</Label>
              <div className="mt-1 max-h-36 overflow-y-auto rounded-lg border border-border/50 bg-input/30 p-2">
                <div className="flex flex-wrap gap-1.5">
                  {symptoms.map((item) => (
                    <button
                      key={item}
                      type="button"
                      onClick={() => toggleItem(selectedSymptoms, item, setSelectedSymptoms)}
                      className={`rounded-md px-2 py-1 text-xs transition-all ${
                        selectedSymptoms.includes(item)
                          ? "bg-primary/20 text-primary border border-primary/30 font-medium"
                          : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
                      }`}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5">BP Systolic</Label>
                <Input type="number" value={systolic} onChange={(event) => setSystolic(Number(event.target.value))} className="bg-input/50" />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5">BP Diastolic</Label>
                <Input type="number" value={diastolic} onChange={(event) => setDiastolic(Number(event.target.value))} className="bg-input/50" />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5">Heart Rate</Label>
                <Input type="number" value={heartRate} onChange={(event) => setHeartRate(Number(event.target.value))} className="bg-input/50" />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5">Temperature (F)</Label>
                <Input type="number" step={0.1} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} className="bg-input/50" />
              </div>
            </div>

            <div>
              <Label className="text-xs text-muted-foreground mb-1.5">Pre-Existing Conditions ({selectedConditions.length})</Label>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {conditions.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => toggleItem(selectedConditions, item, setSelectedConditions)}
                    className={`rounded-md px-2 py-1 text-xs transition-all ${
                      selectedConditions.includes(item)
                        ? "bg-primary/20 text-primary border border-primary/30 font-medium"
                        : "bg-muted/50 text-muted-foreground hover:bg-muted border border-transparent"
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <Button onClick={runDiagnosis} disabled={loading}>
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Run Diagnosis
              </Button>
              <Button onClick={runDiagnosisAndSave} disabled={saving} variant="outline">
                {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Diagnose & Save
              </Button>
            </div>
          </Card>
        </div>

        <div className="xl:col-span-7">
          <Card className="border-border/50 bg-card p-6 space-y-4">
            <h3 className="text-base font-semibold">Diagnosis Output</h3>

            {!prediction ? (
              <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
                Run diagnosis to view differential and recommendations.
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <RiskBadge level={prediction.risk_level || "Low"} size="lg" />
                  <PriorityBadge category={prediction.priority_category || "Standard"} />
                  <span className="rounded-md bg-muted px-2 py-1 text-xs">ESI {prediction.esi_level || "-"}</span>
                  <span className="rounded-md bg-muted px-2 py-1 text-xs">Confidence {((prediction.confidence || 0) * 100).toFixed(1)}%</span>
                </div>

                <div className="rounded-lg border border-border/50 bg-muted/20 p-3 text-sm">
                  <p><span className="text-muted-foreground">Recommended Department:</span> <span className="font-semibold">{prediction.department}</span></p>
                  <p className="mt-1 text-muted-foreground">{prediction.department_reason}</p>
                  <p className="mt-1"><span className="text-muted-foreground">Estimated wait:</span> {prediction.estimated_wait_time}</p>
                </div>

                {Array.isArray(prediction.differential_diagnosis) && prediction.differential_diagnosis.length > 0 ? (
                  <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
                    <p className="text-sm font-semibold">Differential Diagnosis</p>
                    <div className="mt-2 space-y-2">
                      {prediction.differential_diagnosis.map((item, index) => (
                        <div key={index} className="rounded-md border border-border/50 bg-card p-2 text-sm">
                          <p className="font-semibold">{item.diagnosis}</p>
                          <p className="text-xs text-muted-foreground">Likelihood: {item.likelihood}</p>
                          <p className="text-xs text-muted-foreground">{item.rationale}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                  <p className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
                    <ClipboardCheck className="h-4 w-4" />
                    Suggestions / Recommendations
                  </p>
                  <ul className="mt-2 list-disc space-y-1 pl-4 text-sm text-muted-foreground">
                    {recommendations.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>

                {result?.manual_review_recommended || prediction.manual_review_required ? (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
                    <p className="flex items-center gap-2 font-semibold">
                      <ShieldAlert className="h-4 w-4" />
                      Manual Review Alert
                    </p>
                    <p className="mt-1">Model output should be validated by a clinician before final decisions.</p>
                  </div>
                ) : null}

                {result?.clinical_explanation ? (
                  <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3">
                    <p className="text-xs font-semibold text-blue-300">Clinical Explanation</p>
                    <pre className="mt-1 whitespace-pre-wrap text-xs text-muted-foreground">{String(result.clinical_explanation)}</pre>
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
