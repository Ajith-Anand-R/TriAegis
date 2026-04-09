"use client";

import { useEffect, useState } from "react";
import { getCurrentUser, getDashboard } from "@/lib/api";
import { MetricCard } from "@/components/metric-card";
import { Activity, AlertTriangle, BarChart3, Building2, ClipboardList, ClipboardPlus, History, RefreshCw, Stethoscope, Users } from "lucide-react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { useI18n } from "@/components/language-provider";
import { TranslationKey } from "@/lib/i18n";
import { Button } from "@/components/ui/button";
import { DashboardSummary } from "@/lib/types";

const QUICK_LINKS: Array<{ href: string; labelKey: TranslationKey; descKey: TranslationKey; icon: typeof Stethoscope; color: string }> = [
  { href: "/analysis", labelKey: "dashboard.link.analysis", descKey: "dashboard.link.analysisDesc", icon: Stethoscope, color: "from-blue-500/20 to-indigo-500/10" },
  { href: "/diagnosis", labelKey: "dashboard.link.diagnosis", descKey: "dashboard.link.diagnosisDesc", icon: ClipboardPlus, color: "from-fuchsia-500/20 to-pink-500/10" },
  { href: "/queue", labelKey: "dashboard.link.queue", descKey: "dashboard.link.queueDesc", icon: ClipboardList, color: "from-amber-500/20 to-orange-500/10" },
  { href: "/analytics", labelKey: "dashboard.link.analytics", descKey: "dashboard.link.analyticsDesc", icon: BarChart3, color: "from-emerald-500/20 to-green-500/10" },
  { href: "/history", labelKey: "dashboard.link.history", descKey: "dashboard.link.historyDesc", icon: History, color: "from-cyan-500/20 to-blue-500/10" },
];

export default function HomePage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [role, setRole] = useState<string | null>(null);
  const { t } = useI18n();

  const loadDashboard = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getDashboard();
      setSummary(data);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (exc) {
      setSummary(null);
      setError(exc instanceof Error ? exc.message : "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();
    getCurrentUser().then((user) => setRole(user.role)).catch(() => setRole(null));
    const timer = window.setInterval(loadDashboard, 15000);
    return () => window.clearInterval(timer);
  }, []);

  const quickLinks = role === "Admin"
    ? QUICK_LINKS.filter((item) => item.href !== "/analysis")
    : QUICK_LINKS;

  const liveStatus = error || (loading ? "Loading dashboard" : (lastUpdated ? `Dashboard updated at ${lastUpdated}` : ""));

  return (
    <div className="space-y-7">
      <p role="status" aria-live="polite" className="sr-only">{liveStatus}</p>
      <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-blue-500/15 via-card to-cyan-500/10 p-4 sm:p-6">
        <div className="pointer-events-none absolute -right-16 -top-12 h-44 w-44 rounded-full bg-blue-500/20 blur-2xl" />
        <div className="pointer-events-none absolute -bottom-16 -left-10 h-44 w-44 rounded-full bg-cyan-400/15 blur-2xl" />

        <div className="relative flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-cyan-300">
              Operations Hub
            </p>
            <h1 className="font-display mt-1 text-2xl font-semibold tracking-tight md:text-3xl">
              TriAegis {t("dashboard.title")}
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
              {t("dashboard.subtitle")}
            </p>
            <p className="mt-1 text-xs text-muted-foreground/80">
              Source: backend /api/dashboard{lastUpdated ? ` • Last updated ${lastUpdated}` : ""}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-200">
              Auto refresh: 15s
            </span>
            {role ? (
              <span className="rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                Role: {role}
              </span>
            ) : null}
            <Button variant="outline" size="sm" onClick={loadDashboard} disabled={loading} className="min-h-10 border-primary/30 px-3 sm:min-h-9">
              {loading ? <RefreshCw className="mr-1 h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="mr-1 h-3.5 w-3.5" />}
              Refresh
            </Button>
          </div>
        </div>
      </section>

      {/* Metrics */}
      {error ? (
        <Card className="border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          <div className="flex flex-col items-stretch justify-between gap-3 sm:flex-row sm:items-center">
            <span>{error}</span>
            <Button variant="outline" size="sm" onClick={loadDashboard} className="min-h-10 px-3 sm:min-h-9">
              Retry
            </Button>
          </div>
        </Card>
      ) : null}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title={t("dashboard.totalPredictions")}
          value={loading ? "..." : (summary ? summary.total_predictions : "—")}
          icon={Users}
          variant="info"
        />
        <MetricCard
          title={t("dashboard.queueWaiting")}
          value={loading ? "..." : (summary ? summary.queue_waiting : "—")}
          icon={Activity}
          variant="warning"
        />
        <MetricCard
          title={t("dashboard.highRiskCases")}
          value={loading ? "..." : (summary ? summary.high_risk_cases : "—")}
          icon={AlertTriangle}
          variant="danger"
        />
        <MetricCard
          title={t("dashboard.activeDepartments")}
          value={loading ? "..." : (summary ? summary.active_departments : "—")}
          icon={Building2}
          variant="success"
        />
      </div>

      {/* Quick Links */}
      <div>
        <h2 className="font-display mb-4 text-lg font-semibold">{t("dashboard.quickAccess")}</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          {quickLinks.map((link) => (
            <Link key={link.href} href={link.href}>
              <Card className="group relative min-h-[124px] overflow-hidden border-border/50 bg-card p-4 transition-all duration-300 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5 sm:p-5">
                <div className={`absolute inset-0 bg-gradient-to-br ${link.color} opacity-0 transition-opacity group-hover:opacity-100`} />
                <div className="relative flex flex-col gap-3">
                  <link.icon className="h-6 w-6 text-muted-foreground transition-colors group-hover:text-primary" />
                  <div>
                    <p className="font-semibold">{t(link.labelKey)}</p>
                    <p className="text-xs text-muted-foreground">{t(link.descKey)}</p>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
