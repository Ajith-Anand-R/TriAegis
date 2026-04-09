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
    ClipboardPlus,
    ChevronLeft,
    ChevronRight,
    LogIn,
    LogOut,
    Moon,
    Menu,
    Scale,
    Sun,
    Clock3,
    X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getAuthToken, clearAuthToken, getCurrentUser } from "@/lib/api";
import { useEffect, useRef, useState, type ComponentType } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useI18n } from "@/components/language-provider";
import { LanguageSwitcher } from "@/components/language-switcher";
import { useTheme } from "@/components/theme-provider";
import { TranslationKey } from "@/lib/i18n";

const NAV_ITEMS: Array<{ href: string; key: TranslationKey; icon: ComponentType<{ className?: string }> }> = [
    { href: "/", key: "sidebar.dashboard", icon: LayoutDashboard },
    { href: "/analysis", key: "sidebar.singleAnalysis", icon: Stethoscope },
    { href: "/diagnosis", key: "sidebar.diagnosis", icon: ClipboardPlus },
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
    const [isMobile, setIsMobile] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [authReady, setAuthReady] = useState(false);
    const [role, setRole] = useState<string | null>(null);
    const openMenuButtonRef = useRef<HTMLButtonElement | null>(null);
    const closeMenuButtonRef = useRef<HTMLButtonElement | null>(null);
    const sidebarRef = useRef<HTMLElement | null>(null);
    const lastFocusedElementRef = useRef<HTMLElement | null>(null);
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

    useEffect(() => {
        if (typeof window === "undefined") {
            return;
        }

        const media = window.matchMedia("(max-width: 1023px)");
        const syncLayoutMode = () => {
            const mobile = media.matches;
            setIsMobile(mobile);
            if (!mobile) {
                setMobileOpen(false);
            }
        };

        syncLayoutMode();
        media.addEventListener("change", syncLayoutMode);

        return () => {
            media.removeEventListener("change", syncLayoutMode);
        };
    }, []);

    useEffect(() => {
        if (!isMobile || !mobileOpen) {
            return;
        }

        lastFocusedElementRef.current =
            document.activeElement instanceof HTMLElement ? document.activeElement : null;

        const focusCloseButton = window.setTimeout(() => {
            closeMenuButtonRef.current?.focus();
        }, 0);

        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                event.preventDefault();
                setMobileOpen(false);
                return;
            }

            if (event.key !== "Tab") {
                return;
            }

            const focusable = sidebarRef.current?.querySelectorAll<HTMLElement>(
                'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
            );

            if (!focusable || focusable.length === 0) {
                return;
            }

            const focusableElements = Array.from(focusable);
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];

            if (event.shiftKey && document.activeElement === firstElement) {
                event.preventDefault();
                lastElement.focus();
            } else if (!event.shiftKey && document.activeElement === lastElement) {
                event.preventDefault();
                firstElement.focus();
            }
        };

        window.addEventListener("keydown", onKeyDown);
        return () => {
            window.clearTimeout(focusCloseButton);
            window.removeEventListener("keydown", onKeyDown);
            lastFocusedElementRef.current?.focus();
            lastFocusedElementRef.current = null;
        };
    }, [isMobile, mobileOpen]);

    const navItems = role === "Admin"
        ? NAV_ITEMS.filter((item) => item.href !== "/analysis")
        : NAV_ITEMS;

    const handleLogout = () => {
        clearAuthToken();
        router.push("/login");
    };

    const isCollapsed = !isMobile && collapsed;

    const handleNavItemClick = () => {
        if (isMobile) {
            setMobileOpen(false);
        }
    };

    return (
        <>
            {isMobile && !mobileOpen ? (
                <button
                    ref={openMenuButtonRef}
                    type="button"
                    onClick={() => setMobileOpen(true)}
                    className="fixed left-3 top-3 z-[75] flex h-10 w-10 items-center justify-center rounded-lg border border-sidebar-border bg-sidebar text-sidebar-foreground shadow-lg transition-colors hover:bg-sidebar-accent rtl:left-auto rtl:right-3"
                    aria-label="Open navigation menu"
                    aria-expanded={mobileOpen}
                    aria-controls="app-sidebar"
                    aria-haspopup="dialog"
                    title="Open navigation menu"
                >
                    <Menu className="h-5 w-5" />
                </button>
            ) : null}

            {isMobile && mobileOpen ? (
                <button
                    type="button"
                    className="fixed inset-0 z-[69] bg-black/40 backdrop-blur-[1px]"
                    onClick={() => setMobileOpen(false)}
                    aria-label="Close navigation menu"
                />
            ) : null}

            <aside
                ref={sidebarRef}
                id="app-sidebar"
                className={cn(
                    "fixed left-0 top-0 z-[70] flex h-screen flex-col border-r border-sidebar-border bg-sidebar transition-all duration-300 rtl:left-auto rtl:right-0 rtl:border-r-0 rtl:border-l",
                    isMobile
                        ? "w-[280px] shadow-2xl"
                        : isCollapsed
                            ? "w-[68px]"
                            : "w-[240px]",
                    isMobile
                        ? (mobileOpen ? "translate-x-0" : "-translate-x-full rtl:translate-x-full")
                        : "translate-x-0"
                )}
                aria-label="Main navigation"
                role={isMobile && mobileOpen ? "dialog" : "navigation"}
                aria-modal={isMobile && mobileOpen ? true : undefined}
                aria-hidden={isMobile && !mobileOpen ? true : undefined}
            >
            {/* Logo */}
                <div className="flex h-16 items-center gap-3 border-b border-sidebar-border px-4">
                    <Link
                        href="/"
                        className="flex min-w-0 flex-1 items-center gap-3 rounded-md py-1 transition-colors hover:bg-sidebar-accent"
                        onClick={handleNavItemClick}
                    >
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg">
                            <HeartPulse className="h-5 w-5 text-white" />
                        </div>
                        {!isCollapsed && (
                            <div className="flex min-w-0 flex-col overflow-hidden">
                                <span className="truncate text-sm font-bold tracking-tight text-sidebar-foreground">
                                    TriAegis
                                </span>
                                <span className="truncate text-[10px] font-medium text-muted-foreground">
                                    {t("sidebar.subtitle")}
                                </span>
                            </div>
                        )}
                    </Link>

                    {isMobile ? (
                        <button
                            ref={closeMenuButtonRef}
                            type="button"
                            onClick={() => setMobileOpen(false)}
                            className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-foreground"
                            aria-label="Close navigation menu"
                            title="Close navigation menu"
                        >
                            <X className="h-4 w-4" />
                        </button>
                    ) : null}
                </div>

            {/* Navigation */}
            <nav className="flex-1 space-y-1 px-3 py-4">
                {navItems.map((item) => {
                    const active = item.href === "/" ? pathname === "/" : (pathname === item.href || pathname.startsWith(item.href + "/"));
                    const Icon = item.icon;

                    const linkContent = (
                        <Link
                            key={item.href}
                            href={item.href}
                            onClick={handleNavItemClick}
                            aria-label={t(item.key)}
                            aria-current={active ? "page" : undefined}
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
                            {!isCollapsed && <span className="truncate">{t(item.key)}</span>}
                            {active && !isCollapsed && (
                                <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary rtl:ml-0 rtl:mr-auto" />
                            )}
                        </Link>
                    );

                    if (isCollapsed) {
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
                    onClick={handleNavItemClick}
                    aria-label={t("sidebar.health")}
                    className={cn(
                        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground",
                        pathname === "/health" && "bg-primary/15 text-primary"
                    )}
                >
                    <Activity className="h-[18px] w-[18px] shrink-0" />
                    {!isCollapsed && <span>{t("sidebar.health")}</span>}
                </Link>

                {!isCollapsed ? <div className="mt-2"><LanguageSwitcher /></div> : null}

                <button
                    type="button"
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
                    {!isCollapsed && <span>{theme === "dark" ? t("theme.lightMode") : t("theme.darkMode")}</span>}
                </button>

                {authReady && isAuthenticated ? (
                    <button
                        type="button"
                        onClick={() => {
                            handleLogout();
                            handleNavItemClick();
                        }}
                        className="mt-2 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground"
                        aria-label={t("sidebar.logout")}
                    >
                        <LogOut className="h-[18px] w-[18px] shrink-0" />
                        {!isCollapsed && <span>{t("sidebar.logout")}</span>}
                    </button>
                ) : authReady ? (
                    <Link
                        href="/login"
                        onClick={handleNavItemClick}
                        aria-label={t("sidebar.login")}
                        className={cn(
                            "mt-2 flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground",
                            pathname === "/login" && "bg-primary/15 text-primary"
                        )}
                    >
                        <LogIn className="h-[18px] w-[18px] shrink-0" />
                        {!isCollapsed && <span>{t("sidebar.login")}</span>}
                    </Link>
                ) : null}

                {!isMobile ? (
                    <button
                        type="button"
                        onClick={() => setCollapsed(!collapsed)}
                        className="mt-2 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-sidebar-foreground"
                        aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
                        title={collapsed ? "Expand navigation" : t("sidebar.collapse")}
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
                ) : null}
            </div>
            </aside>
        </>
    );
}
