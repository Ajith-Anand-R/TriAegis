"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { clearAuthToken, getCurrentUser, login, setAuthUserRole } from "@/lib/api";
import { Loader2, ShieldCheck } from "lucide-react";
import { useI18n } from "@/components/language-provider";
import { LanguageSwitcher } from "@/components/language-switcher";

export default function LoginPage() {
    const router = useRouter();
    const { t } = useI18n();
    const [username, setUsername] = useState("doctor");
    const [password, setPassword] = useState("doctor123");
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [checking, setChecking] = useState(true);

    useEffect(() => {
        const checkAuth = async () => {
            try {
                await getCurrentUser();
                router.replace("/");
            } catch {
                clearAuthToken();
            } finally {
                setChecking(false);
            }
        };

        checkAuth();
    }, [router]);

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setLoading(true);
        setError(null);
        try {
            await login(username.trim(), password);
            const user = await getCurrentUser();
            setAuthUserRole(user.role);
            router.replace("/");
        } catch (exc: unknown) {
            setError(exc instanceof Error ? exc.message : "Login failed");
        } finally {
            setLoading(false);
        }
    };

    if (checking) {
        return (
            <div className="flex min-h-[70vh] items-center justify-center">
                <Loader2 className="h-7 w-7 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="mx-auto flex min-h-[70vh] max-w-md items-center">
            <Card className="w-full border-border/50 bg-card p-6">
                <div className="mb-4">
                    <LanguageSwitcher />
                </div>
                <div className="mb-5 flex items-center gap-2">
                    <ShieldCheck className="h-5 w-5 text-primary" />
                    <h1 className="text-lg font-semibold">{t("login.title")}</h1>
                </div>

                <form className="space-y-4" onSubmit={handleSubmit}>
                    <div>
                        <Label className="mb-1.5 text-xs text-muted-foreground">{t("login.username")}</Label>
                        <Input
                            value={username}
                            onChange={(event) => setUsername(event.target.value)}
                            autoComplete="username"
                            required
                        />
                    </div>

                    <div>
                        <Label className="mb-1.5 text-xs text-muted-foreground">{t("login.password")}</Label>
                        <Input
                            type="password"
                            value={password}
                            onChange={(event) => setPassword(event.target.value)}
                            autoComplete="current-password"
                            required
                        />
                    </div>

                    {error && <p className="text-sm text-destructive">{error}</p>}

                    <Button type="submit" className="w-full" disabled={loading}>
                        {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                        {t("login.button")}
                    </Button>
                </form>

                <p className="mt-4 text-xs text-muted-foreground">
                    {t("login.demoUsers")}
                </p>
            </Card>
        </div>
    );
}
