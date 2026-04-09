"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  ArrowRight,
  Clock3,
  Loader2,
  ShieldCheck,
  Stethoscope,
  UserPlus,
} from "lucide-react";

import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/components/language-provider";
import { clearAuthToken, getCurrentUser, login, setAuthUserRole } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const { t } = useI18n();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [checkingAuth, setCheckingAuth] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkExistingSession = async () => {
      try {
        await getCurrentUser();
        router.replace("/");
      } catch {
        clearAuthToken();
      } finally {
        setCheckingAuth(false);
      }
    };

    checkExistingSession();
  }, [router]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      await login(username.trim(), password);
      const user = await getCurrentUser();
      setAuthUserRole(user.role);
      router.replace("/");
    } catch (exc: unknown) {
      const message = exc instanceof Error ? exc.message : "Login failed";
      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  if (checkingAuth) {
    return (
      <div className="flex min-h-[78vh] items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="mx-auto grid min-h-[78vh] max-w-5xl items-center gap-6 px-4 sm:px-0 lg:grid-cols-[1.1fr_0.9fr]">
      <p role="status" aria-live="polite" className="sr-only">{error || ""}</p>
      <div className="relative hidden overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-teal-600 via-cyan-700 to-blue-800 p-8 text-white shadow-2xl lg:block">
        <div className="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-white/20 blur-2xl" />
        <div className="pointer-events-none absolute -bottom-20 -left-12 h-60 w-60 rounded-full bg-cyan-200/25 blur-3xl" />

        <div className="relative z-10">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1 text-xs font-medium">
            <ShieldCheck className="h-4 w-4" />
            Secure Triage Access
          </div>

          <h2 className="text-3xl font-semibold leading-tight">
            One dashboard for doctors, nurses, and triage admins.
          </h2>
          <p className="mt-3 max-w-md text-sm text-cyan-100">
            Sign in to run AI-assisted triage, manage admissions, and monitor routing quality in real time.
          </p>

          <div className="mt-8 space-y-3">
            <div className="flex items-center gap-3 rounded-lg bg-white/10 p-3 backdrop-blur-sm">
              <Stethoscope className="h-5 w-5" />
              <span className="text-sm">Faster single-patient risk analysis</span>
            </div>
            <div className="flex items-center gap-3 rounded-lg bg-white/10 p-3 backdrop-blur-sm">
              <Clock3 className="h-5 w-5" />
              <span className="text-sm">Queue-driven admit and discharge workflow</span>
            </div>
            <div className="flex items-center gap-3 rounded-lg bg-white/10 p-3 backdrop-blur-sm">
              <Activity className="h-5 w-5" />
              <span className="text-sm">Operational analytics for route quality</span>
            </div>
          </div>
        </div>
      </div>

      <Card className="w-full border-border/60 bg-card/95 p-6 shadow-xl sm:p-7">
        <div className="mb-6 flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">{t("login.title")}</h1>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit} aria-busy={submitting}>
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

          {error ? <p className="text-sm text-destructive" role="alert">{error}</p> : null}

          <Button type="submit" className="min-h-10 w-full px-3 sm:min-h-9" disabled={submitting}>
            {submitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            {t("login.button")}
          </Button>
        </form>

        <div className="mt-5 rounded-lg border border-border/60 bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Need to add a new doctor or staff account?</p>
          <Link
            href="/login/add-user"
            className="mt-2 inline-flex items-center gap-2 text-sm font-medium text-primary transition-colors hover:text-primary/80"
          >
            <UserPlus className="h-4 w-4" />
            Open Add User
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>

        <p className="mt-4 text-xs text-muted-foreground">
          Demo users: doctor/doctor123, nurse/nurse123, admin/admin123
        </p>
      </Card>
    </div>
  );
}
