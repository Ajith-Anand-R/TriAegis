"use client";

import { useCallback, useEffect, useState } from "react";
import { Scale, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { exportFairnessCsv, getFairness } from "@/lib/api";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useI18n } from "@/components/language-provider";

type MetricMap = {
  demographic_parity_gender: number;
  demographic_parity_age: number;
  equal_opportunity_gender: number;
  equal_opportunity_age: number;
};

type DistributionRow = {
  Gender?: string;
  age_band?: string;
  predicted_risk: "Low" | "Medium" | "High";
  count: number;
};

type RateTableRow = {
  Gender?: string;
  age_band?: string;
  high_risk_rate?: number;
  tpr?: number;
  sample_size: number;
};

type FairnessResponse = {
  metrics: MetricMap;
  distributions: {
    gender: DistributionRow[];
    age_band: DistributionRow[];
  };
  tables: {
    demographic_parity_gender: RateTableRow[];
    demographic_parity_age: RateTableRow[];
    equal_opportunity_gender: RateTableRow[];
    equal_opportunity_age: RateTableRow[];
  };
};

const RISK_COLORS: Record<string, string> = {
  Low: "#22c55e",
  Medium: "#f59e0b",
  High: "#ef4444",
};

const chartTooltipStyle = {
  background: "#1a1a2e",
  border: "1px solid #333",
  borderRadius: 8,
  fontSize: 12,
};

export default function FairnessPage() {
  const { t } = useI18n();
  const [sampleSize, setSampleSize] = useState(2000);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<FairnessResponse | null>(null);
  const [exporting, setExporting] = useState<string | null>(null);
  const [exportNotice, setExportNotice] = useState<string | null>(null);

  const runAnalysis = useCallback(async (background = false) => {
    if (!background) {
      setLoading(true);
      setExportNotice(null);
    }
    setError(null);
    try {
      const result = await getFairness(sampleSize);
      setData(result as unknown as FairnessResponse);
    } catch (exc: unknown) {
      setError(exc instanceof Error ? exc.message : t("fairness.failed"));
    } finally {
      if (!background) {
        setLoading(false);
      }
    }
  }, [sampleSize, t]);

  useEffect(() => {
    runAnalysis();
    const timer = window.setInterval(() => {
      runAnalysis(true);
    }, 15000);
    return () => window.clearInterval(timer);
  }, [runAnalysis]);

  const metrics: MetricMap = data?.metrics ?? {
    demographic_parity_gender: 0,
    demographic_parity_age: 0,
    equal_opportunity_gender: 0,
    equal_opportunity_age: 0,
  };
  const tables: FairnessResponse["tables"] = data?.tables ?? {
    demographic_parity_gender: [],
    demographic_parity_age: [],
    equal_opportunity_gender: [],
    equal_opportunity_age: [],
  };

  const genderDist = data?.distributions.gender ?? [];
  const ageDist = data?.distributions.age_band ?? [];

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

  const downloadReport = async (reportType: string, fileName: string) => {
    setExporting(fileName);
    setExportNotice(null);
    try {
      const blob = await exportFairnessCsv(reportType, sampleSize);
      downloadBlob(blob, fileName);
      setExportNotice(`${t("common.exported")} ${fileName}`);
    } catch (exc: unknown) {
      setError(exc instanceof Error ? exc.message : t("common.csvExportFailed"));
    } finally {
      setExporting(null);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
          <Scale className="h-6 w-6 text-primary" />
          {t("fairness.title")}
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("fairness.subtitle")}
        </p>
      </div>

      <Card className="border-border/50 bg-card p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">{t("fairness.sampleSize")}</p>
            <Input
              type="number"
              min={100}
              max={5000}
              value={sampleSize}
              onChange={(e) => setSampleSize(Number(e.target.value))}
              className="w-40"
            />
          </div>
          <Button onClick={() => { runAnalysis(); }} disabled={loading}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            {t("fairness.run")}
          </Button>
        </div>
        {error ? <p className="mt-2 text-sm text-destructive">{error}</p> : null}
        {exportNotice ? <p className="mt-2 text-sm text-emerald-400">{exportNotice}</p> : null}
        {exporting ? <p className="mt-2 text-sm text-muted-foreground">{t("common.exporting")} {exporting}...</p> : null}
      </Card>

      {data ? (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card className="border-border/50 bg-card p-4">
              <p className="text-xs text-muted-foreground">{t("fairness.dpGender")}</p>
              <p className="mt-1 text-xl font-semibold">{metrics.demographic_parity_gender.toFixed(4)}</p>
            </Card>
            <Card className="border-border/50 bg-card p-4">
              <p className="text-xs text-muted-foreground">{t("fairness.dpAge")}</p>
              <p className="mt-1 text-xl font-semibold">{metrics.demographic_parity_age.toFixed(4)}</p>
            </Card>
            <Card className="border-border/50 bg-card p-4">
              <p className="text-xs text-muted-foreground">{t("fairness.eoGender")}</p>
              <p className="mt-1 text-xl font-semibold">{metrics.equal_opportunity_gender.toFixed(4)}</p>
            </Card>
            <Card className="border-border/50 bg-card p-4">
              <p className="text-xs text-muted-foreground">{t("fairness.eoAge")}</p>
              <p className="mt-1 text-xl font-semibold">{metrics.equal_opportunity_age.toFixed(4)}</p>
            </Card>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card className="border-border/50 bg-card p-4">
              <p className="mb-3 text-sm font-semibold">{t("fairness.distGender")}</p>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={genderDist} margin={{ left: 12, right: 12, top: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.01 260)" />
                  <XAxis dataKey="Gender" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={chartTooltipStyle}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="count" name="Predictions">
                    {genderDist.map((row, idx) => (
                      <Cell key={`${row.Gender}-${idx}`} fill={RISK_COLORS[row.predicted_risk] || "#6366f1"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card className="border-border/50 bg-card p-4">
              <p className="mb-3 text-sm font-semibold">{t("fairness.distAge")}</p>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={ageDist} margin={{ left: 12, right: 12, top: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.01 260)" />
                  <XAxis dataKey="age_band" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={chartTooltipStyle}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="count" name="Predictions">
                    {ageDist.map((row, idx) => (
                      <Cell key={`${row.age_band}-${idx}`} fill={RISK_COLORS[row.predicted_risk] || "#6366f1"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card className="border-border/50 bg-card p-4">
              <p className="mb-3 text-sm font-semibold">{t("fairness.eoByGender")}</p>
              <Button
                variant="outline"
                size="sm"
                className="mb-3"
                onClick={() => downloadReport("equal_opportunity_gender", "equal_opportunity_gender.csv")}
                disabled={!tables.equal_opportunity_gender.length || Boolean(exporting)}
              >
                {t("common.exportCsv")}
              </Button>
              <div className="max-h-72 overflow-auto rounded border border-border/50">
                <table className="w-full text-xs">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-3 py-2 text-left">{t("common.group")}</th>
                      <th className="px-3 py-2 text-left">TPR</th>
                      <th className="px-3 py-2 text-left">{t("common.samples")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.equal_opportunity_gender.map((row, idx) => (
                      <tr key={`eog-${idx}`} className="border-t border-border/40">
                        <td className="px-3 py-2">{row.Gender ?? "-"}</td>
                        <td className="px-3 py-2">{Number(row.tpr ?? 0).toFixed(4)}</td>
                        <td className="px-3 py-2">{row.sample_size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card className="border-border/50 bg-card p-4">
              <p className="mb-3 text-sm font-semibold">{t("fairness.eoByAge")}</p>
              <Button
                variant="outline"
                size="sm"
                className="mb-3"
                onClick={() => downloadReport("equal_opportunity_age", "equal_opportunity_age.csv")}
                disabled={!tables.equal_opportunity_age.length || Boolean(exporting)}
              >
                {t("common.exportCsv")}
              </Button>
              <div className="max-h-72 overflow-auto rounded border border-border/50">
                <table className="w-full text-xs">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-3 py-2 text-left">{t("common.group")}</th>
                      <th className="px-3 py-2 text-left">TPR</th>
                      <th className="px-3 py-2 text-left">{t("common.samples")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.equal_opportunity_age.map((row, idx) => (
                      <tr key={`eoa-${idx}`} className="border-t border-border/40">
                        <td className="px-3 py-2">{row.age_band ?? "-"}</td>
                        <td className="px-3 py-2">{Number(row.tpr ?? 0).toFixed(4)}</td>
                        <td className="px-3 py-2">{row.sample_size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          <Card className="border-border/50 bg-card p-4">
            <p className="mb-3 text-sm font-semibold">{t("fairness.dpTables")}</p>
            <div className="mb-3 flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => downloadReport("demographic_parity_gender", "demographic_parity_gender.csv")}
                disabled={!tables.demographic_parity_gender.length || Boolean(exporting)}
              >
                {t("fairness.exportGenderCsv")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => downloadReport("demographic_parity_age", "demographic_parity_age.csv")}
                disabled={!tables.demographic_parity_age.length || Boolean(exporting)}
              >
                {t("fairness.exportAgeCsv")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => downloadReport("prediction_distribution_gender", "prediction_distribution_gender.csv")}
                disabled={!genderDist.length || Boolean(exporting)}
              >
                {t("fairness.exportDistGender")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => downloadReport("prediction_distribution_age", "prediction_distribution_age.csv")}
                disabled={!ageDist.length || Boolean(exporting)}
              >
                {t("fairness.exportDistAge")}
              </Button>
            </div>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="max-h-72 overflow-auto rounded border border-border/50">
                <table className="w-full text-xs">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-3 py-2 text-left">{t("common.gender")}</th>
                      <th className="px-3 py-2 text-left">{t("analytics.highRiskRate")}</th>
                      <th className="px-3 py-2 text-left">{t("common.samples")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.demographic_parity_gender.map((row, idx) => (
                      <tr key={`dpg-${idx}`} className="border-t border-border/40">
                        <td className="px-3 py-2">{row.Gender ?? "-"}</td>
                        <td className="px-3 py-2">{Number(row.high_risk_rate ?? 0).toFixed(4)}</td>
                        <td className="px-3 py-2">{row.sample_size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="max-h-72 overflow-auto rounded border border-border/50">
                <table className="w-full text-xs">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-3 py-2 text-left">{t("common.ageBand")}</th>
                      <th className="px-3 py-2 text-left">{t("analytics.highRiskRate")}</th>
                      <th className="px-3 py-2 text-left">{t("common.samples")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.demographic_parity_age.map((row, idx) => (
                      <tr key={`dpa-${idx}`} className="border-t border-border/40">
                        <td className="px-3 py-2">{row.age_band ?? "-"}</td>
                        <td className="px-3 py-2">{Number(row.high_risk_rate ?? 0).toFixed(4)}</td>
                        <td className="px-3 py-2">{row.sample_size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Card>
        </>
      ) : (
        <Card className="border-border/50 bg-card p-6 text-sm text-muted-foreground">
          {t("fairness.runHint")}
        </Card>
      )}
    </div>
  );
}
