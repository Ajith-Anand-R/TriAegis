"use client";

import { useEffect, useState, useCallback } from "react";
import { getAnalytics } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { MetricCard } from "@/components/metric-card";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Cell,
    PieChart,
    Pie,
    Legend,
    LineChart,
    Line,
    CartesianGrid,
} from "recharts";
import {
    BarChart3,
    Loader2,
    Activity,
    AlertTriangle,
    CalendarDays,
    RefreshCw,
    Target,
    Shield,
    SlidersHorizontal,
    Users,
    Zap,
    TrendingUp,
    Route,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";

const RISK_COLORS: Record<string, string> = {
    High: "#ef4444",
    Medium: "#f59e0b",
    Low: "#22c55e",
};

const DEPT_COLORS = [
    "#0ea5e9",
    "#22c55e",
    "#f59e0b",
    "#ef4444",
    "#14b8a6",
    "#3b82f6",
    "#f97316",
    "#84cc16",
    "#eab308",
    "#06b6d4",
];

const tooltipStyle = {
    background: "#1a1a2e",
    border: "1px solid #333",
    borderRadius: 8,
    fontSize: 12,
};

export default function AnalyticsPage() {
    const { t } = useI18n();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [startDate, setStartDate] = useState("");
    const [endDate, setEndDate] = useState("");
    const [riskFilter, setRiskFilter] = useState<string[]>([]);

    const fetchData = useCallback(async (background = false) => {
        if (!background) {
            setLoading(true);
        }
        try {
            const filters: Record<string, string> = {};
            if (startDate) filters.start_date = startDate;
            if (endDate) filters.end_date = endDate;
            if (riskFilter.length) filters.risk_levels = riskFilter.join(",");
            const d = await getAnalytics(filters);
            setData(d);
        } catch {
            /* ignore */
        } finally {
            if (!background) {
                setLoading(false);
            }
        }
    }, [startDate, endDate, riskFilter]);

    useEffect(() => {
        fetchData();
        const timer = window.setInterval(() => {
            fetchData(true);
        }, 10000);
        return () => window.clearInterval(timer);
    }, [fetchData]);

    if (loading) {
        return (
            <div className="flex h-[60vh] items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    if (!data || data.empty) {
        return (
            <div className="flex h-[60vh] flex-col items-center justify-center text-center text-muted-foreground">
                <BarChart3 className="mb-3 h-12 w-12 opacity-40" />
                <p className="text-lg font-medium">{t("analytics.noData")}</p>
                <p className="text-sm">{t("analytics.noDataHint")}</p>
            </div>
        );
    }

    const m = data.metrics || {};
    const charts = data.charts || {};
    const routingQuality = data.routing_quality || {};

    const routingTotalRoutes = Number(m.routing_total_routes ?? routingQuality.total_routes ?? 0);
    const routingCapacityHitRate = Number(m.routing_capacity_hit_rate ?? routingQuality.capacity_hit_rate ?? 0);
    const routingOverflowRate = Number(m.routing_overflow_rate ?? routingQuality.overflow_rate ?? 0);
    const routingWaitDelta = Number(m.routing_mean_wait_delta ?? routingQuality.mean_wait_delta_minutes ?? 0);
    const activeFilterCount = (startDate ? 1 : 0) + (endDate ? 1 : 0) + (riskFilter.length > 0 ? 1 : 0);

    const riskPieData = Object.entries(charts.risk_counts || {}).map(([name, value]) => ({ name, value }));
    const deptBarData = Object.entries(charts.dept_counts || {}).map(([name, value]) => ({ name, value }));
    const symptomData = Object.entries(charts.symptoms || {}).map(([name, value]) => ({ name, value }));
    const recentActivity = Array.isArray(data.recent_activity) ? data.recent_activity : [];

    // Transform trend data for line chart
    const trendDates = [...new Set((charts.trend || []).map((t: { date: string }) => t.date))].sort();
    const trendLineData = trendDates.map((date) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const row: any = { date };
        for (const rl of ["High", "Medium", "Low"]) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const found = (charts.trend || []).find((t: any) => t.date === date && t.risk_level === rl);
            row[rl] = found ? found.count : 0;
        }
        return row;
    });

    return (
        <div className="space-y-7">
            <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-cyan-500/15 via-card to-blue-500/10 p-4 sm:p-6">
                <div className="pointer-events-none absolute -right-14 -top-12 h-40 w-40 rounded-full bg-cyan-500/20 blur-2xl" />
                <div className="pointer-events-none absolute -bottom-16 -left-8 h-44 w-44 rounded-full bg-blue-500/15 blur-2xl" />

                <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div>
                        <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-cyan-300">
                            Operational Intelligence
                        </p>
                        <h1 className="font-display mt-1 flex items-center gap-2 text-2xl font-semibold tracking-tight md:text-3xl">
                            <BarChart3 className="h-6 w-6 text-primary" />
                            {t("analytics.title")}
                        </h1>
                        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
                            {t("analytics.subtitle")}
                        </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-200">
                            Auto refresh: 10s
                        </span>
                        <span className="rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                            Active filters: {activeFilterCount}
                        </span>
                        <Button size="sm" variant="outline" className="min-h-10 border-primary/30 px-3 sm:min-h-9" onClick={() => { fetchData(true); }}>
                            <RefreshCw className="mr-1 h-3.5 w-3.5" />
                            Refresh
                        </Button>
                    </div>
                </div>
            </section>

            <Card className="border-border/60 bg-gradient-to-r from-card to-card/90 p-4 shadow-sm">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                        <SlidersHorizontal className="h-4 w-4 text-primary" />
                        <p className="font-display text-sm font-semibold">Analytics Filters</p>
                    </div>
                    <Button variant="ghost" size="sm" className="min-h-10 px-3 sm:min-h-9" onClick={() => { setStartDate(""); setEndDate(""); setRiskFilter([]); }}>
                        {t("analytics.reset")}
                    </Button>
                </div>

                <div className="flex flex-wrap items-end gap-4">
                    <div className="w-full sm:w-auto">
                        <Label className="mb-1.5 flex items-center gap-1 text-xs text-muted-foreground">
                            <CalendarDays className="h-3 w-3" />
                            {t("analytics.startDate")}
                        </Label>
                        <Input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="h-10 w-full bg-input/50 sm:h-9 sm:w-44"
                        />
                    </div>
                    <div className="w-full sm:w-auto">
                        <Label className="mb-1.5 flex items-center gap-1 text-xs text-muted-foreground">
                            <CalendarDays className="h-3 w-3" />
                            {t("analytics.endDate")}
                        </Label>
                        <Input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="h-10 w-full bg-input/50 sm:h-9 sm:w-44"
                        />
                    </div>
                    <div className="flex w-full flex-wrap gap-1.5 sm:w-auto">
                        {["High", "Medium", "Low"].map((rl) => {
                            const selected = riskFilter.includes(rl);
                            const activeClass =
                                rl === "High"
                                    ? "border-red-500/30 bg-red-500/10 text-red-300"
                                    : rl === "Medium"
                                        ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                                        : "border-emerald-500/30 bg-emerald-500/10 text-emerald-300";

                            return (
                                <Button
                                    key={rl}
                                    size="sm"
                                    variant="outline"
                                    onClick={() => setRiskFilter((prev) =>
                                        prev.includes(rl) ? prev.filter((r) => r !== rl) : [...prev, rl]
                                    )}
                                    className={`${selected ? activeClass : "text-muted-foreground"} min-h-10 px-3 sm:min-h-9`}
                                    aria-pressed={selected}
                                >
                                    {rl}
                                </Button>
                            );
                        })}
                    </div>
                </div>
            </Card>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
                <MetricCard title={t("analytics.predictions")} value={m.total_predictions} icon={Users} variant="info" />
                <MetricCard title={t("analytics.highRiskRate")} value={`${m.high_risk_rate}%`} icon={AlertTriangle} variant="danger" />
                <MetricCard title={t("analytics.avgPriority")} value={m.avg_priority} icon={Target} variant="warning" />
                <MetricCard title={t("analytics.modelConfidence")} value={`${m.avg_confidence}%`} icon={Shield} variant="success" />
                <MetricCard title={t("analytics.queueWaiting")} value={m.queue_waiting} icon={Activity} variant="info" />
            </div>

            <Card className="border-border/60 bg-gradient-to-r from-card to-card/90 p-4 shadow-sm">
                <div className="mb-3 flex items-center gap-2">
                    <Route className="h-4 w-4 text-primary" />
                    <p className="font-display text-sm font-semibold">Routing Quality (24h)</p>
                </div>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <MetricCard title="Routing Decisions" value={routingTotalRoutes} icon={Route} variant="info" />
                    <MetricCard title="Capacity Hit Rate" value={`${routingCapacityHitRate.toFixed(1)}%`} icon={Shield} variant="success" />
                    <MetricCard title="Overflow Rate" value={`${routingOverflowRate.toFixed(1)}%`} icon={AlertTriangle} variant="danger" />
                    <MetricCard title="Mean Wait Delta" value={`${routingWaitDelta.toFixed(1)} min`} icon={TrendingUp} variant={routingWaitDelta > 0 ? "warning" : "success"} />
                </div>
            </Card>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                    <h3 className="font-display mb-4 flex items-center gap-2 text-sm font-semibold">
                        <AlertTriangle className="h-4 w-4 text-muted-foreground" /> {t("analytics.riskDistribution")}
                    </h3>
                    <ResponsiveContainer width="100%" height={240}>
                        <PieChart>
                            <Pie data={riskPieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={90} strokeWidth={2} stroke="oklch(0.13 0.01 260)">
                                {riskPieData.map((entry) => (
                                    <Cell key={entry.name} fill={RISK_COLORS[entry.name as string] || "#0ea5e9"} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={tooltipStyle} />
                            <Legend verticalAlign="bottom" iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                        </PieChart>
                    </ResponsiveContainer>
                </Card>

                <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                    <h3 className="font-display mb-4 flex items-center gap-2 text-sm font-semibold">
                        <Zap className="h-4 w-4 text-muted-foreground" /> {t("analytics.departmentLoad")}
                    </h3>
                    <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={deptBarData} margin={{ left: 0, right: 0, bottom: 30 }}>
                            <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#888" }} angle={-35} textAnchor="end" interval={0} />
                            <YAxis tick={{ fontSize: 10, fill: "#888" }} />
                            <Tooltip contentStyle={tooltipStyle} />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                {deptBarData.map((_, i) => (
                                    <Cell key={i} fill={DEPT_COLORS[i % DEPT_COLORS.length]} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </Card>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {trendLineData.length > 0 ? (
                    <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                        <h3 className="font-display mb-4 flex items-center gap-2 text-sm font-semibold">
                            <TrendingUp className="h-4 w-4 text-muted-foreground" /> {t("analytics.riskTrend")}
                        </h3>
                        <ResponsiveContainer width="100%" height={240}>
                            <LineChart data={trendLineData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.01 260)" />
                                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#888" }} />
                                <YAxis tick={{ fontSize: 10, fill: "#888" }} />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend wrapperStyle={{ fontSize: 12 }} />
                                <Line type="monotone" dataKey="High" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                                <Line type="monotone" dataKey="Medium" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                                <Line type="monotone" dataKey="Low" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </Card>
                ) : (
                    <Card className="flex min-h-[320px] items-center justify-center border-border/60 bg-gradient-to-b from-card to-card/90 p-4 text-sm text-muted-foreground shadow-sm sm:p-6">
                        Not enough trend data for the selected filters.
                    </Card>
                )}

                <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                    <h3 className="font-display mb-4 flex items-center gap-2 text-sm font-semibold">
                        <Activity className="h-4 w-4 text-muted-foreground" /> {t("analytics.topSymptoms")}
                    </h3>
                    <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={symptomData} layout="vertical" margin={{ left: 10 }}>
                            <XAxis type="number" tick={{ fontSize: 10, fill: "#888" }} />
                            <YAxis type="category" dataKey="name" tick={{ fontSize: 9, fill: "#ccc" }} width={130} />
                            <Tooltip contentStyle={tooltipStyle} />
                            <Bar dataKey="value" fill="#0ea5e9" radius={[0, 4, 4, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </Card>
            </div>

            {recentActivity.length > 0 && (
                <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                    <div className="mb-4 flex items-center justify-between gap-2">
                        <h3 className="font-display text-sm font-semibold">{t("analytics.recentActivity")}</h3>
                        <span className="rounded-full border border-border/60 bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground">
                            {recentActivity.length} records
                        </span>
                    </div>
                    <div className="space-y-2 md:hidden">
                        {recentActivity.map((row: {
                            timestamp?: string;
                            patient_id?: string;
                            risk_level?: string;
                            priority_score?: number;
                            recommended_department?: string;
                        }, i: number) => (
                            <article key={`recent-mobile-${i}`} className="rounded-lg border border-border/60 bg-card/70 p-3">
                                <div className="flex items-center justify-between gap-2">
                                    <p className="font-mono text-xs font-semibold">{row.patient_id}</p>
                                    <span className={`text-xs font-semibold ${row.risk_level === "High" ? "text-red-400" : row.risk_level === "Medium" ? "text-amber-400" : "text-emerald-400"}`}>
                                        {row.risk_level}
                                    </span>
                                </div>
                                <p className="mt-1 text-[11px] text-muted-foreground">{row.timestamp?.split(".")[0]?.replace("T", " ")}</p>
                                <div className="mt-2 flex items-center justify-between text-xs">
                                    <span className="text-muted-foreground">Score: <span className="font-semibold text-foreground">{Number(row.priority_score).toFixed(1)}</span></span>
                                    <span className="text-muted-foreground">{row.recommended_department}</span>
                                </div>
                            </article>
                        ))}
                    </div>

                    <div className="hidden md:block">
                        <div className="mobile-scroll max-h-[320px] overflow-auto rounded-lg border border-border/50">
                            <table className="w-full min-w-[720px] text-sm">
                                <caption className="sr-only">Recent triage activity table</caption>
                                <thead className="sticky top-0 bg-muted/85 backdrop-blur">
                                    <tr className="border-b border-border/50">
                                        <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Time</th>
                                        <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Patient</th>
                                        <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Risk</th>
                                        <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Score</th>
                                        <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Department</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                    {recentActivity.map((row: any, i: number) => (
                                        <tr key={i} className="border-b border-border/30 hover:bg-muted/20">
                                            <td className="px-3 py-2 text-xs text-muted-foreground">{row.timestamp?.split(".")[0]?.replace("T", " ")}</td>
                                            <td className="px-3 py-2 font-mono text-xs">{row.patient_id}</td>
                                            <td className="px-3 py-2">
                                                <span className={`text-xs font-semibold ${row.risk_level === "High" ? "text-red-400" : row.risk_level === "Medium" ? "text-amber-400" : "text-emerald-400"}`}>
                                                    {row.risk_level}
                                                </span>
                                            </td>
                                            <td className="px-3 py-2 font-semibold">{Number(row.priority_score).toFixed(1)}</td>
                                            <td className="px-3 py-2 text-muted-foreground">{row.recommended_department}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </Card>
            )}
        </div>
    );
}
