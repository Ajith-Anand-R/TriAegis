"use client";

import { useCallback, useEffect, useState } from "react";
import { Clock3, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RiskBadge } from "@/components/risk-badge";
import { exportSimulationCsv, getSimulationState, resetSimulation, stepSimulation } from "@/lib/api";
import { useI18n } from "@/components/language-provider";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type SimState = {
  current_minute: number;
  queue: SimQueueRow[];
  timeline: SimTimelineRow[];
  arrived: number;
  completed: number;
};

type SimQueueRow = {
  patient_id: string;
  age: number;
  risk_level: string;
  priority_score: number;
  department: string;
  arrival_minute: number;
  queue_position: number;
};

type SimTimelineRow = {
  minute: number;
  department: string;
  waiting: number;
};

const EMPTY_STATE: SimState = {
  current_minute: 0,
  queue: [],
  timeline: [],
  arrived: 0,
  completed: 0,
};

const chartTooltipStyle = {
  background: "#1a1a2e",
  border: "1px solid #333",
  borderRadius: 8,
  fontSize: 12,
};

export default function SimulationPage() {
  const { t } = useI18n();
  const [minutes, setMinutes] = useState(15);
  const [lambdaRate, setLambdaRate] = useState(1.5);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);
  const [state, setState] = useState<SimState>(EMPTY_STATE);

  const loadState = useCallback(async () => {
    try {
      const data = await getSimulationState();
      setState(data as unknown as SimState);
    } catch {
      setState(EMPTY_STATE);
    }
  }, []);

  useEffect(() => {
    loadState();
    const timer = window.setInterval(() => {
      loadState();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [loadState]);

  const runStep = async () => {
    setLoading(true);
    setError(null);
    setExportNotice(null);
    try {
      const data = await stepSimulation(minutes, lambdaRate, seed);
      setState(data as unknown as SimState);
    } catch (exc: unknown) {
      setError(exc instanceof Error ? exc.message : t("simulation.stepFailed"));
    } finally {
      setLoading(false);
    }
  };

  const reset = async () => {
    setLoading(true);
    setError(null);
    setExportNotice(null);
    try {
      const data = await resetSimulation();
      setState(data as unknown as SimState);
    } catch (exc: unknown) {
      setError(exc instanceof Error ? exc.message : t("simulation.resetFailed"));
    } finally {
      setLoading(false);
    }
  };

  const downloadBlob = (blob: Blob, fileName: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", fileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const downloadReport = async (reportType: "queue" | "timeline", fileName: string) => {
    setExporting(fileName);
    setExportNotice(null);
    try {
      const blob = await exportSimulationCsv(reportType);
      downloadBlob(blob, fileName);
      setExportNotice(`${t("common.exported")} ${fileName}`);
    } catch (exc: unknown) {
      setError(exc instanceof Error ? exc.message : t("common.csvExportFailed"));
    } finally {
      setExporting(null);
    }
  };

  const timelineByMinute: Array<Record<string, number | string>> = [];
  const grouped: Record<number, Record<string, number | string>> = {};
  for (const row of state.timeline) {
    const existing = grouped[row.minute] || { minute: row.minute };
    existing[row.department] = row.waiting;
    grouped[row.minute] = existing;
  }
  for (const minute of Object.keys(grouped).map((x) => Number(x)).sort((a, b) => a - b)) {
    timelineByMinute.push(grouped[minute]);
  }

  const deptNames = Array.from(new Set(state.timeline.map((row) => row.department)));
  const deptColors = [
    "#ef4444",
    "#f59e0b",
    "#22c55e",
    "#3b82f6",
    "#8b5cf6",
    "#06b6d4",
    "#a855f7",
    "#10b981",
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
          <Clock3 className="h-6 w-6 text-primary" />
          {t("simulation.title")}
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("simulation.subtitle")}
        </p>
      </div>

      <Card className="border-border/50 bg-card p-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">{t("simulation.minutes")}</p>
            <Input type="number" min={1} max={120} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} />
          </div>
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">{t("simulation.arrivals")}</p>
            <Input type="number" step={0.1} min={0.2} max={4} value={lambdaRate} onChange={(e) => setLambdaRate(Number(e.target.value))} />
          </div>
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">{t("simulation.seed")}</p>
            <Input type="number" min={1} max={999999} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </div>
          <div className="flex items-end gap-2">
            <Button className="flex-1" onClick={runStep} disabled={loading || Boolean(exporting)}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              {t("simulation.runStep")}
            </Button>
            <Button variant="outline" onClick={reset} disabled={loading || Boolean(exporting)}>{t("simulation.reset")}</Button>
          </div>
        </div>
        {error ? <p className="mt-2 text-sm text-destructive">{error}</p> : null}
        {exportNotice ? <p className="mt-2 text-sm text-emerald-400">{exportNotice}</p> : null}
        {exporting ? <p className="mt-2 text-sm text-muted-foreground">{t("common.exporting")} {exporting}...</p> : null}
      </Card>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Card className="border-border/50 bg-card p-4"><p className="text-xs text-muted-foreground">{t("simulation.simMinute")}</p><p className="text-xl font-semibold">{state.current_minute}</p></Card>
        <Card className="border-border/50 bg-card p-4"><p className="text-xs text-muted-foreground">{t("simulation.arrived")}</p><p className="text-xl font-semibold">{state.arrived}</p></Card>
        <Card className="border-border/50 bg-card p-4"><p className="text-xs text-muted-foreground">{t("simulation.completed")}</p><p className="text-xl font-semibold">{state.completed}</p></Card>
        <Card className="border-border/50 bg-card p-4"><p className="text-xs text-muted-foreground">{t("simulation.waiting")}</p><p className="text-xl font-semibold">{state.queue.length}</p></Card>
      </div>

      <Card className="border-border/50 bg-card p-4">
        <p className="mb-3 text-sm font-semibold">{t("simulation.currentQueue")}</p>
        <Button
          variant="outline"
          size="sm"
          className="mb-3"
          onClick={() => downloadReport("queue", "simulation_queue_snapshot.csv")}
          disabled={!state.queue.length || Boolean(exporting)}
        >
          {t("simulation.exportQueue")}
        </Button>
        {state.queue.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("simulation.emptyQueue")}</p>
        ) : (
          <div className="max-h-80 overflow-auto rounded border border-border/50">
            <table className="w-full text-xs">
              <thead className="bg-muted/40">
                <tr>
                  <th className="px-3 py-2 text-left">{t("common.pos")}</th>
                  <th className="px-3 py-2 text-left">{t("common.patient")}</th>
                  <th className="px-3 py-2 text-left">{t("common.risk")}</th>
                  <th className="px-3 py-2 text-left">{t("common.priority")}</th>
                  <th className="px-3 py-2 text-left">{t("common.department")}</th>
                  <th className="px-3 py-2 text-left">{t("simulation.arrived")}</th>
                </tr>
              </thead>
              <tbody>
                {state.queue.map((row) => (
                  <tr key={row.patient_id} className="border-t border-border/40">
                    <td className="px-3 py-2">{row.queue_position}</td>
                    <td className="px-3 py-2 font-mono">{row.patient_id}</td>
                    <td className="px-3 py-2"><RiskBadge level={row.risk_level} size="sm" /></td>
                    <td className="px-3 py-2">{row.priority_score.toFixed(1)}</td>
                    <td className="px-3 py-2">{row.department}</td>
                    <td className="px-3 py-2">{row.arrival_minute}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card className="border-border/50 bg-card p-4">
        <p className="mb-3 text-sm font-semibold">{t("simulation.departmentLoad")}</p>
        <Button
          variant="outline"
          size="sm"
          className="mb-3"
          onClick={() => downloadReport("timeline", "simulation_department_timeline.csv")}
          disabled={!state.timeline.length || Boolean(exporting)}
        >
          {t("simulation.exportTimeline")}
        </Button>
        {timelineByMinute.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("simulation.runToGenerate")}</p>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={timelineByMinute} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.01 260)" />
              <XAxis dataKey="minute" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {deptNames.map((dept, idx) => (
                <Line
                  key={dept}
                  type="monotone"
                  dataKey={dept}
                  stroke={deptColors[idx % deptColors.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>
    </div>
  );
}
