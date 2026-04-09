"use client";

import { useEffect, useState, useCallback, useMemo, type ElementType, type FormEvent } from "react";
import {
    changePassword,
    getAuthUsers,
    getCurrentUser,
    getHealthcheck,
    registerUser,
    updateUserStatus,
} from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
    ShieldCheck,
    UserPlus,
} from "lucide-react";
import { useI18n } from "@/components/language-provider";

type HealthPayload = {
    ok: boolean;
    checks: Record<string, string>;
    details: Record<string, unknown>;
};

type AuthUserRow = {
    username: string;
    role: string;
    is_active: boolean;
};

const ICONS: Record<string, ElementType> = {
    model: Brain,
    preprocessor: Brain,
    database: Database,
    data: FileText,
};


function normalizeHealthPayload(raw: Record<string, unknown>): HealthPayload {
    const checksRaw = raw["checks"];
    const normalizedChecks: Record<string, string> = {};

    if (checksRaw && typeof checksRaw === "object") {
        for (const [key, value] of Object.entries(checksRaw)) {
            normalizedChecks[key] = String(value);
        }
    }

    const detailsRaw = raw["details"];
    const normalizedDetails = detailsRaw && typeof detailsRaw === "object"
        ? (detailsRaw as Record<string, unknown>)
        : {};

    return {
        ok: Boolean(raw["ok"]),
        checks: normalizedChecks,
        details: normalizedDetails,
    };
}

function formatDetailValue(value: unknown): string {
    if (value === null || value === undefined) {
        return "N/A";
    }

    if (typeof value === "object") {
        try {
            return JSON.stringify(value);
        } catch {
            return String(value);
        }
    }

    return String(value);
}

export default function HealthPage() {
    const { t } = useI18n();
    const [health, setHealth] = useState<HealthPayload | null>(null);
    const [loading, setLoading] = useState(true);

    const [role, setRole] = useState<string | null>(null);
    const [currentUsername, setCurrentUsername] = useState<string | null>(null);
    const [users, setUsers] = useState<AuthUserRow[]>([]);

    const [currentPassword, setCurrentPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [passwordLoading, setPasswordLoading] = useState(false);
    const [passwordMessage, setPasswordMessage] = useState<string | null>(null);
    const [passwordError, setPasswordError] = useState<string | null>(null);

    const [createUsername, setCreateUsername] = useState("");
    const [createPassword, setCreatePassword] = useState("");
    const [createRole, setCreateRole] = useState("Nurse");
    const [createLoading, setCreateLoading] = useState(false);
    const [createMessage, setCreateMessage] = useState<string | null>(null);
    const [createError, setCreateError] = useState<string | null>(null);

    const [statusLoadingUser, setStatusLoadingUser] = useState<string | null>(null);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [statusError, setStatusError] = useState<string | null>(null);
    const [userFilter, setUserFilter] = useState("");

    const fetch = useCallback(async (background = false) => {
        if (!background) {
            setLoading(true);
        }
        try {
            const d = await getHealthcheck();
            setHealth(normalizeHealthPayload(d));
        } catch {
            setHealth({ ok: false, checks: {}, details: {} });
        } finally {
            if (!background) {
                setLoading(false);
            }
        }
    }, []);

    const fetchUsers = useCallback(async () => {
        try {
            const response = await getAuthUsers();
            setUsers(response.users || []);
        } catch {
            setUsers([]);
        }
    }, []);

    const loadAuthContext = useCallback(async () => {
        try {
            const me = await getCurrentUser();
            setRole(me.role);
            setCurrentUsername(me.username);
            if (me.role === "Admin") {
                await fetchUsers();
            } else {
                setUsers([]);
            }
        } catch {
            setRole(null);
            setCurrentUsername(null);
            setUsers([]);
        }
    }, [fetchUsers]);

    useEffect(() => {
        fetch();
        loadAuthContext();

        const timer = window.setInterval(() => {
            fetch(true);
        }, 10000);

        return () => window.clearInterval(timer);
    }, [fetch, loadAuthContext]);

    const handlePasswordChange = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        setPasswordMessage(null);
        setPasswordError(null);

        if (newPassword !== confirmPassword) {
            setPasswordError("New password and confirmation do not match");
            return;
        }

        setPasswordLoading(true);
        try {
            const response = await changePassword({
                current_password: currentPassword,
                new_password: newPassword,
            });
            setPasswordMessage(response.message || "Password updated successfully");
            setCurrentPassword("");
            setNewPassword("");
            setConfirmPassword("");
        } catch (exc: unknown) {
            const message = exc instanceof Error ? exc.message : "Password change failed";
            setPasswordError(message);
        } finally {
            setPasswordLoading(false);
        }
    };

    const handleCreateUser = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setCreateError(null);
        setCreateMessage(null);

        const username = createUsername.trim().toLowerCase();
        if (!username) {
            setCreateError("Username is required");
            return;
        }

        setCreateLoading(true);
        try {
            const response = await registerUser({
                username,
                password: createPassword,
                role: createRole,
            });
            setCreateMessage(`User ${response.created.username} created`);
            setCreateUsername("");
            setCreatePassword("");
            await fetchUsers();
        } catch (exc: unknown) {
            const message = exc instanceof Error ? exc.message : "User creation failed";
            setCreateError(message);
        } finally {
            setCreateLoading(false);
        }
    };

    const handleToggleUserStatus = async (username: string, isActive: boolean) => {
        setStatusMessage(null);
        setStatusError(null);
        setStatusLoadingUser(username);

        try {
            const response = await updateUserStatus({
                username,
                is_active: !isActive,
            });

            setStatusMessage(
                response.updated.is_active
                    ? `User ${response.updated.username} enabled`
                    : `User ${response.updated.username} disabled`
            );

            await fetchUsers();
        } catch (exc: unknown) {
            const message = exc instanceof Error ? exc.message : "User status update failed";
            setStatusError(message);
        } finally {
            setStatusLoadingUser(null);
        }
    };

    const checks = health?.checks || {};
    const allKeys = Object.keys(checks);
    const detailEntries = useMemo(() => Object.entries(health?.details || {}), [health?.details]);
    const filteredUsers = useMemo(() => {
        const q = userFilter.trim().toLowerCase();
        if (!q) {
            return users;
        }

        return users.filter((item) => (
            item.username.toLowerCase().includes(q) || item.role.toLowerCase().includes(q)
        ));
    }, [users, userFilter]);

    const liveStatus = passwordError || passwordMessage || createError || createMessage || statusError || statusMessage || "";

    return (
        <div className="space-y-7" aria-busy={loading || passwordLoading || createLoading || Boolean(statusLoadingUser)}>
            <p role="status" aria-live="polite" className="sr-only">{liveStatus}</p>
            <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-emerald-500/15 via-card to-cyan-500/10 p-4 sm:p-6">
                <div className="pointer-events-none absolute -right-12 -top-10 h-36 w-36 rounded-full bg-emerald-500/20 blur-2xl" />
                <div className="pointer-events-none absolute -bottom-12 -left-8 h-44 w-44 rounded-full bg-cyan-400/15 blur-2xl" />

                <div className="relative flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                    <div>
                        <p className="font-display text-xs font-semibold uppercase tracking-[0.18em] text-emerald-300">
                            Runtime Observability
                        </p>
                        <h1 className="font-display mt-1 flex items-center gap-2 text-2xl font-semibold tracking-tight md:text-3xl">
                            <Activity className="h-6 w-6 text-primary" />
                            {t("health.title")}
                        </h1>
                        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
                            {t("health.subtitle")}
                        </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
                            Auto refresh: 10s
                        </span>
                        <Button variant="outline" size="sm" onClick={() => { fetch(); }} disabled={loading} className="min-h-10 border-primary/30 px-3 sm:min-h-9">
                            {loading ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <RefreshCw className="mr-1 h-3 w-3" />}
                            {t("health.refresh")}
                        </Button>
                    </div>
                </div>
            </section>

            {loading ? (
                <div className="flex h-48 items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
            ) : (
                <>
                    {/* Overall Status */}
                    <Card className={`border p-4 shadow-sm sm:p-6 ${health?.ok ? "border-emerald-500/30 bg-emerald-500/5" : "border-red-500/35 bg-red-500/5"}`}>
                        <div className="flex flex-wrap items-center gap-4">
                            <div className={`rounded-xl p-3 ${health?.ok ? "bg-emerald-500/15" : "bg-red-500/15"}`}>
                                {health?.ok ? (
                                    <CheckCircle2 className="h-8 w-8 text-emerald-400" />
                                ) : (
                                    <XCircle className="h-8 w-8 text-red-400" />
                                )}
                            </div>
                            <div>
                                <p className="font-display text-xl font-semibold">
                                    {health?.ok ? t("health.ok") : t("health.issues")}
                                </p>
                                <p className="text-sm text-muted-foreground">
                                    {allKeys.filter((k) => checks[k] === "ok").length}/{allKeys.length} {t("health.checksPassing")}
                                </p>
                            </div>

                            <div className="ml-auto rounded-lg border border-border/60 bg-card/60 px-3 py-1.5 text-xs font-medium text-muted-foreground">
                                {health?.ok ? "System state: Stable" : "System state: Attention needed"}
                            </div>
                        </div>
                    </Card>

                    {/* Individual Checks */}
                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                        {allKeys.map((key) => {
                            const ok = checks[key] === "ok";
                            const Icon = ICONS[key] || Server;
                            return (
                                <Card key={key} className="border-border/60 bg-gradient-to-b from-card to-card/90 p-5 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md">
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

                    <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                        <div className="mb-4 flex items-center gap-2">
                            <Server className="h-5 w-5 text-primary" />
                            <h2 className="font-display text-lg font-semibold">System Details</h2>
                        </div>

                        {detailEntries.length === 0 ? (
                            <p className="text-sm text-muted-foreground">No additional runtime details were reported by the backend.</p>
                        ) : (
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
                                {detailEntries.map(([key, value]) => (
                                    <div key={key} className="rounded-lg border border-border/60 bg-muted/20 p-3">
                                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{key.replace(/_/g, " ")}</p>
                                        <p className="mt-1 break-words text-sm text-foreground">{formatDetailValue(value)}</p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </Card>

                    <Card className="border-border/60 bg-gradient-to-b from-card to-card/90 p-4 shadow-sm sm:p-6">
                        <div className="mb-4 flex items-center gap-2">
                            <ShieldCheck className="h-5 w-5 text-primary" />
                            <h2 className="font-display text-lg font-semibold">Account and Access</h2>
                        </div>

                        <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
                            <span className="text-muted-foreground">Logged in role:</span>
                            <span className="rounded-full border border-primary/30 bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">
                                {role || "Unknown"}
                            </span>
                        </div>

                        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                            <form className="space-y-3 rounded-xl border border-border/60 bg-muted/20 p-4" onSubmit={handlePasswordChange}>
                                <h3 className="text-sm font-semibold">Change Password</h3>

                                <div>
                                    <Label className="mb-1.5 text-xs text-muted-foreground">Current Password</Label>
                                    <Input
                                        type="password"
                                        value={currentPassword}
                                        onChange={(event) => setCurrentPassword(event.target.value)}
                                        autoComplete="current-password"
                                        required
                                    />
                                </div>

                                <div>
                                    <Label className="mb-1.5 text-xs text-muted-foreground">New Password</Label>
                                    <Input
                                        type="password"
                                        value={newPassword}
                                        onChange={(event) => setNewPassword(event.target.value)}
                                        autoComplete="new-password"
                                        required
                                    />
                                </div>

                                <div>
                                    <Label className="mb-1.5 text-xs text-muted-foreground">Confirm New Password</Label>
                                    <Input
                                        type="password"
                                        value={confirmPassword}
                                        onChange={(event) => setConfirmPassword(event.target.value)}
                                        autoComplete="new-password"
                                        required
                                    />
                                </div>

                                {passwordError ? <p className="text-sm text-destructive" role="alert">{passwordError}</p> : null}
                                {passwordMessage ? <p className="text-sm text-emerald-400" role="status" aria-live="polite">{passwordMessage}</p> : null}

                                <Button type="submit" disabled={passwordLoading} className="min-h-10 border-primary/30 px-3 sm:min-h-9">
                                    {passwordLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                    Update Password
                                </Button>
                            </form>

                            {role === "Admin" ? (
                                <div className="space-y-4 rounded-xl border border-border/60 bg-muted/20 p-4">
                                    <form className="space-y-3" onSubmit={handleCreateUser}>
                                        <div className="flex items-center gap-2">
                                            <UserPlus className="h-4 w-4 text-primary" />
                                            <h3 className="text-sm font-semibold">Create User</h3>
                                        </div>

                                        <div>
                                            <Label className="mb-1.5 text-xs text-muted-foreground">Username</Label>
                                            <Input
                                                value={createUsername}
                                                onChange={(event) => setCreateUsername(event.target.value)}
                                                autoComplete="off"
                                                required
                                            />
                                        </div>

                                        <div>
                                            <Label className="mb-1.5 text-xs text-muted-foreground">Temporary Password</Label>
                                            <Input
                                                type="password"
                                                value={createPassword}
                                                onChange={(event) => setCreatePassword(event.target.value)}
                                                autoComplete="new-password"
                                                required
                                            />
                                        </div>

                                        <div>
                                            <Label className="mb-1.5 text-xs text-muted-foreground">Role</Label>
                                            <select
                                                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                                                value={createRole}
                                                onChange={(event) => setCreateRole(event.target.value)}
                                            >
                                                <option value="Doctor">Doctor</option>
                                                <option value="Nurse">Nurse</option>
                                                <option value="Admin">Admin</option>
                                            </select>
                                        </div>

                                        {createError ? <p className="text-sm text-destructive" role="alert">{createError}</p> : null}
                                        {createMessage ? <p className="text-sm text-emerald-400" role="status" aria-live="polite">{createMessage}</p> : null}

                                        <Button type="submit" disabled={createLoading} className="min-h-10 px-3 sm:min-h-9">
                                            {createLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                            Create Account
                                        </Button>
                                    </form>

                                    <div className="space-y-2">
                                        <div className="flex flex-wrap items-center justify-between gap-2">
                                            <h3 className="text-sm font-semibold">Users</h3>
                                            <Input
                                                value={userFilter}
                                                onChange={(event) => setUserFilter(event.target.value)}
                                                placeholder="Filter by username or role"
                                                className="h-10 w-full max-w-xs sm:h-9"
                                                aria-label="Filter users by username or role"
                                            />
                                        </div>

                                        {statusError ? <p className="text-sm text-destructive" role="alert">{statusError}</p> : null}
                                        {statusMessage ? <p className="text-sm text-emerald-400" role="status" aria-live="polite">{statusMessage}</p> : null}

                                        <div className="space-y-2 md:hidden">
                                            {filteredUsers.length === 0 ? (
                                                <div className="rounded-lg border border-border/60 bg-card/70 px-3 py-3 text-sm text-muted-foreground">
                                                    {users.length === 0 ? "No users available" : "No users match this filter"}
                                                </div>
                                            ) : (
                                                filteredUsers.map((item) => {
                                                    const isCurrentActive = item.username === currentUsername && item.is_active;
                                                    return (
                                                        <article key={`user-mobile-${item.username}`} className="rounded-lg border border-border/60 bg-card/70 p-3">
                                                            <div className="flex items-center justify-between gap-2">
                                                                <p className="font-mono text-xs font-semibold">{item.username}</p>
                                                                <span className={`text-xs ${item.is_active ? "text-emerald-400" : "text-amber-300"}`}>
                                                                    {item.is_active ? "Active" : "Disabled"}
                                                                </span>
                                                            </div>
                                                            <p className="mt-1 text-xs text-muted-foreground">Role: {item.role}</p>
                                                            <Button
                                                                size="sm"
                                                                variant={item.is_active ? "destructive" : "secondary"}
                                                                disabled={statusLoadingUser === item.username || isCurrentActive}
                                                                onClick={() => { handleToggleUserStatus(item.username, item.is_active); }}
                                                                className="mt-2 min-h-10 w-full px-3 sm:min-h-9"
                                                            >
                                                                {statusLoadingUser === item.username ? (
                                                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                                ) : null}
                                                                {isCurrentActive ? "Current User" : item.is_active ? "Disable" : "Enable"}
                                                            </Button>
                                                        </article>
                                                    );
                                                })
                                            )}
                                        </div>

                                        <div className="hidden md:block">
                                            <div className="mobile-scroll overflow-x-auto rounded-lg border border-border/60 bg-card/70">
                                                <table className="w-full min-w-[560px] text-sm">
                                                    <caption className="sr-only">User management table</caption>
                                                    <thead>
                                                        <tr className="border-b border-border/60 bg-muted/30 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                                            <th className="px-3 py-2">Username</th>
                                                            <th className="px-3 py-2">Role</th>
                                                            <th className="px-3 py-2">Status</th>
                                                            <th className="px-3 py-2">Actions</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {filteredUsers.length === 0 ? (
                                                            <tr>
                                                                <td colSpan={4} className="px-3 py-3 text-sm text-muted-foreground">
                                                                    {users.length === 0 ? "No users available" : "No users match this filter"}
                                                                </td>
                                                            </tr>
                                                        ) : (
                                                            filteredUsers.map((item) => {
                                                                const isCurrentActive = item.username === currentUsername && item.is_active;
                                                                return (
                                                                    <tr key={item.username} className="border-b border-border/40 last:border-0">
                                                                        <td className="px-3 py-2 font-mono text-xs sm:text-sm">{item.username}</td>
                                                                        <td className="px-3 py-2">{item.role}</td>
                                                                        <td className={`px-3 py-2 ${item.is_active ? "text-emerald-400" : "text-amber-300"}`}>
                                                                            {item.is_active ? "Active" : "Disabled"}
                                                                        </td>
                                                                        <td className="px-3 py-2">
                                                                            <Button
                                                                                size="sm"
                                                                                variant={item.is_active ? "destructive" : "secondary"}
                                                                                disabled={statusLoadingUser === item.username || isCurrentActive}
                                                                                onClick={() => { handleToggleUserStatus(item.username, item.is_active); }}
                                                                                className="min-h-10 px-3 sm:min-h-9"
                                                                            >
                                                                                {statusLoadingUser === item.username ? (
                                                                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                                                ) : null}
                                                                                {isCurrentActive ? "Current User" : item.is_active ? "Disable" : "Enable"}
                                                                            </Button>
                                                                        </td>
                                                                    </tr>
                                                                );
                                                            })
                                                        )}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="rounded-xl border border-border/60 bg-muted/20 p-4 text-sm text-muted-foreground">
                                    User provisioning is available only for Admin role.
                                </div>
                            )}
                        </div>
                    </Card>
                </>
            )}
        </div>
    );
}
