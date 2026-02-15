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
    Target,
    Shield,
    Users,
    Zap,
    TrendingUp,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";

const RISK_COLORS: Record<string, string> = {
    High: "#ef4444",
    Medium: "#f59e0b",
    Low: "#22c55e",
};

const DEPT_COLORS = [
    "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
    "#818cf8", "#7c3aed", "#5b21b6", "#4c1d95",
    "#3b82f6", "#2563eb",
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

    const riskPieData = Object.entries(charts.risk_counts || {}).map(([name, value]) => ({ name, value }));
    const deptBarData = Object.entries(charts.dept_counts || {}).map(([name, value]) => ({ name, value }));
    const symptomData = Object.entries(charts.symptoms || {}).map(([name, value]) => ({ name, value }));

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
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
                        <BarChart3 className="h-6 w-6 text-primary" />
                        {t("analytics.title")}
                    </h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        {t("analytics.subtitle")}
                    </p>
                </div>
            </div>

            {/* Filters */}
            <Card className="border-border/50 bg-card p-4">
                <div className="flex flex-wrap items-end gap-4">
                    <div>
                        <Label className="text-xs text-muted-foreground mb-1.5">{t("analytics.startDate")}</Label>
                        <Input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="w-40 bg-input/50"
                        />
                    </div>
                    <div>
                        <Label className="text-xs text-muted-foreground mb-1.5">{t("analytics.endDate")}</Label>
                        <Input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-40 bg-input/50"
                        />
                    </div>
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
                    <Button variant="ghost" size="sm" onClick={() => { setStartDate(""); setEndDate(""); setRiskFilter([]); }}>
                        {t("analytics.reset")}
                    </Button>
                </div>
            </Card>

            {/* Metrics Row */}
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
                <MetricCard title={t("analytics.predictions")} value={m.total_predictions} icon={Users} variant="info" />
                <MetricCard title={t("analytics.highRiskRate")} value={`${m.high_risk_rate}%`} icon={AlertTriangle} variant="danger" />
                <MetricCard title={t("analytics.avgPriority")} value={m.avg_priority} icon={Target} variant="warning" />
                <MetricCard title={t("analytics.modelConfidence")} value={`${m.avg_confidence}%`} icon={Shield} variant="success" />
                <MetricCard title={t("analytics.queueWaiting")} value={m.queue_waiting} icon={Activity} variant="info" />
            </div>

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {/* Risk Distribution */}
                <Card className="border-border/50 bg-card p-6">
                    <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold">
                        <AlertTriangle className="h-4 w-4 text-muted-foreground" /> {t("analytics.riskDistribution")}
                    </h3>
                    <ResponsiveContainer width="100%" height={240}>
                        <PieChart>
                            <Pie data={riskPieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={90} strokeWidth={2} stroke="oklch(0.13 0.01 260)">
                                {riskPieData.map((entry) => (
                                    <Cell key={entry.name} fill={RISK_COLORS[entry.name as string] || "#6366f1"} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={tooltipStyle} />
                            <Legend verticalAlign="bottom" iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                        </PieChart>
                    </ResponsiveContainer>
                </Card>

                {/* Department Load */}
                <Card className="border-border/50 bg-card p-6">
                    <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold">
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

            {/* Charts Row 2 */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {/* Risk Trend */}
                {trendLineData.length > 0 && (
                    <Card className="border-border/50 bg-card p-6">
                        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold">
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
                )}

                {/* Top Symptoms */}
                <Card className="border-border/50 bg-card p-6">
                    <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold">
                        <Activity className="h-4 w-4 text-muted-foreground" /> {t("analytics.topSymptoms")}
                    </h3>
                    <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={symptomData} layout="vertical" margin={{ left: 10 }}>
                            <XAxis type="number" tick={{ fontSize: 10, fill: "#888" }} />
                            <YAxis type="category" dataKey="name" tick={{ fontSize: 9, fill: "#ccc" }} width={130} />
                            <Tooltip contentStyle={tooltipStyle} />
                            <Bar dataKey="value" fill="#818cf8" radius={[0, 4, 4, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </Card>
            </div>

            {/* Recent Activity */}
            {data.recent_activity?.length > 0 && (
                <Card className="border-border/50 bg-card p-6">
                    <h3 className="mb-4 text-sm font-semibold">{t("analytics.recentActivity")}</h3>
                    <div className="max-h-[300px] overflow-auto rounded-lg border border-border/50">
                        <table className="w-full text-sm">
                            <thead className="sticky top-0 bg-muted/80 backdrop-blur">
                                <tr className="border-b border-border/50">
                                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Time</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Patient</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Risk</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Score</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Department</th>
                                </tr>
                            </thead>
                            <tbody>
                                {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                                {data.recent_activity.map((row: any, i: number) => (
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
                </Card>
            )}
        </div>
    );
}
