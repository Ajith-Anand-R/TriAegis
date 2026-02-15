"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
    Activity,
    BarChart3,
    ClipboardList,
    History,
    HeartPulse,
    LayoutDashboard,
    Stethoscope,
    ChevronLeft,
    ChevronRight,
    LogIn,
    LogOut,
    Moon,
    Scale,
    Sun,
    Clock3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getAuthToken, clearAuthToken, getCurrentUser } from "@/lib/api";
import { useEffect, useState, type ComponentType } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useI18n } from "@/components/language-provider";
import { LanguageSwitcher } from "@/components/language-switcher";
import { useTheme } from "@/components/theme-provider";
import { TranslationKey } from "@/lib/i18n";

const NAV_ITEMS: Array<{ href: string; key: TranslationKey; icon: ComponentType<{ className?: string }> }> = [
    { href: "/", key: "sidebar.dashboard", icon: LayoutDashboard },
    { href: "/analysis", key: "sidebar.singleAnalysis", icon: Stethoscope },
    { href: "/queue", key: "sidebar.queue", icon: ClipboardList },
    { href: "/analytics", key: "sidebar.analytics", icon: BarChart3 },
    { href: "/history", key: "sidebar.history", icon: History },
    { href: "/fairness", key: "sidebar.fairness", icon: Scale },
    { href: "/simulation", key: "sidebar.simulation", icon: Clock3 },
];

export function Sidebar() {
    const pathname = usePathname();
    const router = useRouter();
    const [collapsed, setCollapsed] = useState(false);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [authReady, setAuthReady] = useState(false);
    const [role, setRole] = useState<string | null>(null);
    const { t } = useI18n();
    const { theme, toggleTheme } = useTheme();
    const nextThemeLabel = theme === "dark" ? t("theme.switchToLight") : t("theme.switchToDark");

    useEffect(() => {
        const initAuth = async () => {
            const hasToken = Boolean(getAuthToken());
            setIsAuthenticated(hasToken);
            if (hasToken) {
                try {
                    const me = await getCurrentUser();
                    setRole(me.role);
                } catch {
                    setRole(null);
                }
            } else {
                setRole(null);
            }
            setAuthReady(true);
        };

        initAuth();
    }, []);

    useEffect(() => {
        if (role === "Admin" && pathname.startsWith("/analysis")) {
            router.replace("/queue");
        }
    }, [role, pathname, router]);

    const navItems = role === "Admin"
        ? NAV_ITEMS.filter((item) => item.href !== "/analysis")
        : NAV_ITEMS;

    const handleLogout = () => {
        clearAuthToken();
        router.push("/login");
    };

    return (
        <aside
            className={cn(
                "fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-sidebar-border bg-sidebar transition-all duration-300 rtl:left-auto rtl:right-0 rtl:border-r-0 rtl:border-l",
                collapsed ? "w-[68px]" : "w-[240px]"
            )}
        >
            {/* Logo */}
            <Link href="/" className="flex h-16 items-center gap-3 border-b border-sidebar-border px-4 transition-colors hover:bg-sidebar-accent">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg">
                    <HeartPulse className="h-5 w-5 text-white" />
                </div>
                {!collapsed && (
                    <div className="flex flex-col overflow-hidden">
                        <span className="text-sm font-bold tracking-tight text-sidebar-foreground">
                            TriAegis
                        </span>
                        <span className="text-[10px] font-medium text-muted-foreground">
                            {t("sidebar.subtitle")}
                        </span>
                    </div>
                )}
            </Link>

            {/* Navigation */}
            <nav className="flex-1 space-y-1 px-3 py-4">
                {navItems.map((item) => {
                    const active = item.href === "/" ? pathname === "/" : (pathname === item.href || pathname.startsWith(item.href + "/"));
                    const Icon = item.icon;

                    const linkContent = (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                                active
                                    ? "bg-primary/15 text-primary shadow-sm"
                                    : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground"
                            )}
                        >
                            <Icon
                                className={cn(
                                    "h-[18px] w-[18px] shrink-0 transition-colors",
                                    active
                                        ? "text-primary"
                                        : "text-muted-foreground group-hover:text-sidebar-foreground"
                                )}
                            />
                            {!collapsed && <span className="truncate">{t(item.key)}</span>}
                            {active && !collapsed && (
                                <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary rtl:ml-0 rtl:mr-auto" />
                            )}
                        </Link>
                    );

                    if (collapsed) {
                        return (
                            <Tooltip key={item.href} delayDuration={0}>
                                <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                                <TooltipContent side="right" className="font-medium rtl:[&]:data-[side=right]:left-auto rtl:[&]:data-[side=right]:right-full">
                                    {t(item.key)}
                                </TooltipContent>
                            </Tooltip>
                        );
                    }

                    return linkContent;
                })}
            </nav>

            {/* Health + Collapse */}
            <div className="border-t border-sidebar-border p-3">
                <Link
                    href="/health"
                    className={cn(
                        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground",
                        pathname === "/health" && "bg-primary/15 text-primary"
                    )}
                >
                    <Activity className="h-[18px] w-[18px] shrink-0" />
                    {!collapsed && <span>{t("sidebar.health")}</span>}
                </Link>

                {!collapsed ? <div className="mt-2"><LanguageSwitcher /></div> : null}

                <button
                    onClick={toggleTheme}
                    className="mt-2 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground"
                    aria-label={nextThemeLabel}
                    title={nextThemeLabel}
                >
                    {theme === "dark" ? (
                        <Sun className="h-[18px] w-[18px] shrink-0" />
                    ) : (
                        <Moon className="h-[18px] w-[18px] shrink-0" />
                    )}
                    {!collapsed && <span>{theme === "dark" ? t("theme.lightMode") : t("theme.darkMode")}</span>}
                </button>

                {authReady && isAuthenticated ? (
                    <button
                        onClick={handleLogout}
                        className="mt-2 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground"
                    >
                        <LogOut className="h-[18px] w-[18px] shrink-0" />
                        {!collapsed && <span>{t("sidebar.logout")}</span>}
                    </button>
                ) : authReady ? (
                    <Link
                        href="/login"
                        className={cn(
                            "mt-2 flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground",
                            pathname === "/login" && "bg-primary/15 text-primary"
                        )}
                    >
                        <LogIn className="h-[18px] w-[18px] shrink-0" />
                        {!collapsed && <span>{t("sidebar.login")}</span>}
                    </Link>
                ) : null}

                <button
                    onClick={() => setCollapsed(!collapsed)}
                    className="mt-2 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground"
                >
                    {collapsed ? (
                        <ChevronRight className="h-[18px] w-[18px] shrink-0" />
                    ) : (
                        <>
                            <ChevronLeft className="h-[18px] w-[18px] shrink-0" />
                            <span>{t("sidebar.collapse")}</span>
                        </>
                    )}
                </button>
            </div>
        </aside>
    );
}
