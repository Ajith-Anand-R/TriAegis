"use client";

import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    icon?: LucideIcon;
    trend?: "up" | "down" | "neutral";
    variant?: "default" | "success" | "warning" | "danger" | "info";
}

const VARIANT_STYLES: Record<string, { bg: string; icon: string; glow: string }> = {
    default: { bg: "from-blue-500/10 to-indigo-500/5", icon: "text-blue-400", glow: "" },
    success: { bg: "from-emerald-500/10 to-green-500/5", icon: "text-emerald-400", glow: "glow-green" },
    warning: { bg: "from-amber-500/10 to-orange-500/5", icon: "text-amber-400", glow: "glow-orange" },
    danger: { bg: "from-red-500/10 to-rose-500/5", icon: "text-red-400", glow: "glow-red" },
    info: { bg: "from-cyan-500/10 to-blue-500/5", icon: "text-cyan-400", glow: "" },
};

export function MetricCard({ title, value, subtitle, icon: Icon, variant = "default" }: MetricCardProps) {
    const style = VARIANT_STYLES[variant];

    return (
        <div
            className={cn(
                "group relative overflow-hidden rounded-xl border border-border/50 bg-card p-5 transition-all duration-300 hover:border-border hover:shadow-lg",
                style.glow
            )}
        >
            <div className={cn("absolute inset-0 bg-gradient-to-br opacity-50", style.bg)} />
            <div className="relative flex items-start justify-between">
                <div className="space-y-1">
                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        {title}
                    </p>
                    <p className="text-2xl font-bold tracking-tight">{value}</p>
                    {subtitle && (
                        <p className="text-xs text-muted-foreground">{subtitle}</p>
                    )}
                </div>
                {Icon && (
                    <div className={cn("rounded-lg bg-background/50 p-2.5", style.icon)}>
                        <Icon className="h-5 w-5" />
                    </div>
                )}
            </div>
        </div>
    );
}
