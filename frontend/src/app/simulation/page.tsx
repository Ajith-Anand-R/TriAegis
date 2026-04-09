"use client";

import { useCallback, useEffect, useState } from "react";
import { Clock3, Download, Loader2, RefreshCw, SlidersHorizontal } from "lucide-react";
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

  const liveStatus = error || exportNotice || (exporting ? `${t("common.exporting")} ${exporting}` : "");

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
    <div className="space-y-7" aria-busy={loading || Boolean(exporting)}>
      <p role="status" aria-live="polite" className="sr-only">{liveStatus}</p>
      <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-sky-500/15 via-card to-indigo-500/10 p-4 sm:p-6">
        <div className="pointer-events-none absolute -right-16 -top-14 h-44 w-44 rounded-full bg-sky-500/20 blur-2xl" />
        <div className="pointer-events-none absolute -bottom-16 -left-10 h-44 w-44 rounded-full bg-indigo-400/15 blur-2xl" />

        <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-sky-300">
              Flow Simulation
            </p>
            <h1 className="font-display mt-1 flex items-center gap-2 text-2xl font-semibold tracking-tight md:text-3xl">
              <Clock3 className="h-6 w-6 text-primary" />
              {t("simulation.title")}
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
              {t("simulation.subtitle")}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-sky-500/30 bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-200">
              Auto refresh: 5s
            </span>
            <span className="rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
              Sim minute: {state.current_minute}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => { loadState(); }}
              disabled={loading || Boolean(exporting)}
              className="min-h-10 border-primary/30 px-3 sm:min-h-9"
            >
              <RefreshCw className="mr-1 h-3.5 w-3.5" />
              Refresh
            </Button>
          </div>
        </div>
      </section>

      <Card className="border-border/60 bg-gradient-to-r from-card to-card/90 p-4 shadow-sm">
        <div className="mb-3 flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-primary" />
          <p className="font-display text-sm font-semibold">Simulation Controls</p>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">{t("simulation.minutes")}</p>
            <Input type="number" min={1} max={120} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} className="h-10 sm:h-9" />
          </div>
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">{t("simulation.arrivals")}</p>
            <Input type="number" step={0.1} min={0.2} max={4} value={lambdaRate} onChange={(e) => setLambdaRate(Number(e.target.value))} className="h-10 sm:h-9" />
          </div>
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">{t("simulation.seed")}</p>
            <Input type="number" min={1} max={999999} value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="h-10 sm:h-9" />
          </div>
          <div className="flex flex-col items-stretch justify-end gap-2 sm:flex-row">
            <Button className="min-h-10 flex-1 px-3 sm:min-h-9" onClick={runStep} disabled={loading || Boolean(exporting)}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              {t("simulation.runStep")}
            </Button>
            <Button variant="outline" onClick={reset} disabled={loading || Boolean(exporting)} className="min-h-10 px-3 sm:min-h-9">{t("simulation.reset")}</Button>
          </div>
        </div>
        {error ? <p className="mt-2 text-sm text-destructive" role="alert">{error}</p> : null}
        {exportNotice ? <p className="mt-2 text-sm text-emerald-400" role="status" aria-live="polite">{exportNotice}</p> : null}
        {exporting ? <p className="mt-2 text-sm text-muted-foreground" role="status" aria-live="polite">{t("common.exporting")} {exporting}...</p> : null}
      </Card>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-border/60 bg-card p-4 shadow-sm"><p className="text-xs text-muted-foreground">{t("simulation.simMinute")}</p><p className="text-xl font-semibold">{state.current_minute}</p></Card>
        <Card className="border-border/60 bg-card p-4 shadow-sm"><p className="text-xs text-muted-foreground">{t("simulation.arrived")}</p><p className="text-xl font-semibold">{state.arrived}</p></Card>
        <Card className="border-border/60 bg-card p-4 shadow-sm"><p className="text-xs text-muted-foreground">{t("simulation.completed")}</p><p className="text-xl font-semibold">{state.completed}</p></Card>
        <Card className="border-border/60 bg-card p-4 shadow-sm"><p className="text-xs text-muted-foreground">{t("simulation.waiting")}</p><p className="text-xl font-semibold">{state.queue.length}</p></Card>
      </div>

      <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm">
        <p className="font-display mb-3 text-sm font-semibold">{t("simulation.currentQueue")}</p>
        <Button
          variant="outline"
          size="sm"
          className="mb-3 min-h-10 px-3 sm:min-h-9"
          onClick={() => downloadReport("queue", "simulation_queue_snapshot.csv")}
          disabled={!state.queue.length || Boolean(exporting)}
        >
          <Download className="mr-1 h-3.5 w-3.5" />
          {t("simulation.exportQueue")}
        </Button>
        {state.queue.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("simulation.emptyQueue")}</p>
        ) : (
          <>
            <div className="space-y-2 md:hidden">
              {state.queue.map((row) => (
                <article key={`sim-mobile-${row.patient_id}`} className="rounded-lg border border-border/60 bg-card/70 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-mono text-xs font-semibold">{row.patient_id}</p>
                    <RiskBadge level={row.risk_level} size="sm" />
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-1 text-xs text-muted-foreground">
                    <p>{t("common.pos")}: {row.queue_position}</p>
                    <p>{t("common.priority")}: {row.priority_score.toFixed(1)}</p>
                    <p>{t("common.department")}: {row.department}</p>
                    <p>{t("simulation.arrived")}: {row.arrival_minute}</p>
                  </div>
                </article>
              ))}
            </div>

            <div className="hidden md:block">
              <div className="mobile-scroll max-h-80 overflow-auto rounded border border-border/50">
                <table className="w-full text-xs">
                  <caption className="sr-only">Simulation queue table</caption>
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
            </div>
          </>
        )}
      </Card>

      <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm">
        <p className="font-display mb-3 text-sm font-semibold">{t("simulation.departmentLoad")}</p>
        <Button
          variant="outline"
          size="sm"
          className="mb-3 min-h-10 px-3 sm:min-h-9"
          onClick={() => downloadReport("timeline", "simulation_department_timeline.csv")}
          disabled={!state.timeline.length || Boolean(exporting)}
        >
          <Download className="mr-1 h-3.5 w-3.5" />
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
