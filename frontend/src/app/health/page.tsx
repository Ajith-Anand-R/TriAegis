"use client";

import { useEffect, useState, useCallback } from "react";
import { getHealthcheck } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
    Activity,
    RefreshCw,
    CheckCircle2,
    XCircle,
    Loader2,
    Server,
    Database,
    Brain,
    FileText,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";

const ICONS: Record<string, React.ElementType> = {
    model: Brain,
    preprocessor: Brain,
    database: Database,
    data: FileText,
};

export default function HealthPage() {
    const { t } = useI18n();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [health, setHealth] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    const fetch = useCallback(async (background = false) => {
        if (!background) {
            setLoading(true);
        }
        try {
            const d = await getHealthcheck();
            setHealth(d);
        } catch {
            setHealth({ ok: false, checks: {}, details: {} });
        } finally {
            if (!background) {
                setLoading(false);
            }
        }
    }, []);

    useEffect(() => {
        fetch();
        const timer = window.setInterval(() => {
            fetch(true);
        }, 10000);
        return () => window.clearInterval(timer);
    }, [fetch]);

    const checks = health?.checks || {};
    const allKeys = Object.keys(checks);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
                        <Activity className="h-6 w-6 text-primary" />
                        {t("health.title")}
                    </h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        {t("health.subtitle")}
                    </p>
                </div>
                <Button variant="outline" size="sm" onClick={() => { fetch(); }} disabled={loading}>
                    {loading ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <RefreshCw className="mr-1 h-3 w-3" />}
                    {t("health.refresh")}
                </Button>
            </div>

            {loading ? (
                <div className="flex h-48 items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
            ) : (
                <>
                    {/* Overall Status */}
                    <Card className={`border-border/50 p-6 ${health?.ok ? "glow-green" : "glow-red"}`}>
                        <div className="flex items-center gap-4">
                            <div className={`rounded-full p-3 ${health?.ok ? "bg-emerald-500/15" : "bg-red-500/15"}`}>
                                {health?.ok ? (
                                    <CheckCircle2 className="h-8 w-8 text-emerald-400" />
                                ) : (
                                    <XCircle className="h-8 w-8 text-red-400" />
                                )}
                            </div>
                            <div>
                                <p className="text-xl font-bold">
                                    {health?.ok ? t("health.ok") : t("health.issues")}
                                </p>
                                <p className="text-sm text-muted-foreground">
                                    {allKeys.filter((k) => checks[k] === "ok").length}/{allKeys.length} {t("health.checksPassing")}
                                </p>
                            </div>
                        </div>
                    </Card>

                    {/* Individual Checks */}
                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                        {allKeys.map((key) => {
                            const ok = checks[key] === "ok";
                            const Icon = ICONS[key] || Server;
                            return (
                                <Card key={key} className="border-border/50 bg-card p-5 transition-all hover:shadow-md">
                                    <div className="flex items-center gap-3">
                                        <div className={`rounded-lg p-2.5 ${ok ? "bg-emerald-500/10" : "bg-red-500/10"}`}>
                                            <Icon className={`h-5 w-5 ${ok ? "text-emerald-400" : "text-red-400"}`} />
                                        </div>
                                        <div className="flex-1">
                                            <p className="text-sm font-semibold capitalize">{key.replace(/_/g, " ")}</p>
                                            <p className={`text-xs font-medium ${ok ? "text-emerald-400" : "text-red-400"}`}>
                                                {checks[key]}
                                            </p>
                                        </div>
                                        {ok ? (
                                            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                                        ) : (
                                            <XCircle className="h-5 w-5 text-red-400" />
                                        )}
                                    </div>
                                </Card>
                            );
                        })}
                    </div>
                </>
            )}
        </div>
    );
}
