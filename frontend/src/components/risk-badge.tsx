"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Zap, CheckCircle2 } from "lucide-react";

interface RiskBadgeProps {
    level: "Low" | "Medium" | "High" | string;
    size?: "sm" | "md" | "lg";
}

const CONFIG: Record<string, { bg: string; text: string; border: string; icon: React.ElementType; label: string }> = {
    High: {
        bg: "bg-red-500/15",
        text: "text-red-400",
        border: "border-red-500/30",
        icon: AlertTriangle,
        label: "HIGH RISK",
    },
    Medium: {
        bg: "bg-amber-500/15",
        text: "text-amber-400",
        border: "border-amber-500/30",
        icon: Zap,
        label: "MEDIUM RISK",
    },
    Low: {
        bg: "bg-emerald-500/15",
        text: "text-emerald-400",
        border: "border-emerald-500/30",
        icon: CheckCircle2,
        label: "LOW RISK",
    },
};

const SIZE_STYLES = {
    sm: "px-2.5 py-1 text-xs gap-1.5",
    md: "px-3.5 py-2 text-sm gap-2",
    lg: "px-5 py-3 text-base gap-2.5 font-bold",
};

export function RiskBadge({ level, size = "md" }: RiskBadgeProps) {
    const cfg = CONFIG[level] || CONFIG.Low;
    const Icon = cfg.icon;

    return (
        <div
            className={cn(
                "inline-flex items-center rounded-lg border font-semibold transition-all",
                cfg.bg,
                cfg.text,
                cfg.border,
                SIZE_STYLES[size]
            )}
        >
            <Icon className={cn(size === "sm" ? "h-3 w-3" : size === "lg" ? "h-5 w-5" : "h-4 w-4")} />
            {cfg.label}
        </div>
    );
}

export function PriorityBadge({ category }: { category: string }) {
    const colorMap: Record<string, string> = {
        Critical: "bg-red-500/15 text-red-400 border-red-500/30",
        Urgent: "bg-orange-500/15 text-orange-400 border-orange-500/30",
        High: "bg-amber-500/15 text-amber-400 border-amber-500/30",
        Standard: "bg-blue-500/15 text-blue-400 border-blue-500/30",
        Low: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    };

    return (
        <Badge
            variant="outline"
            className={cn("font-semibold", colorMap[category] || colorMap.Standard)}
        >
            {category}
        </Badge>
    );
}

export function StatusBadge({ status }: { status: string }) {
    const colorMap: Record<string, string> = {
        waiting: "bg-amber-500/15 text-amber-400 border-amber-500/30",
        in_progress: "bg-blue-500/15 text-blue-400 border-blue-500/30",
        completed: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    };

    const labels: Record<string, string> = {
        waiting: "Waiting",
        in_progress: "In Progress",
        completed: "Completed",
    };

    return (
        <Badge
            variant="outline"
            className={cn("font-medium", colorMap[status] || colorMap.waiting)}
        >
            {labels[status] || status}
        </Badge>
    );
}
