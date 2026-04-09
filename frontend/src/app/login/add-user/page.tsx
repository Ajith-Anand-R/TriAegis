"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { ArrowLeft, Loader2, UserPlus, UserRoundCog } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { registerUserWithAdminCredentials } from "@/lib/api";

export default function AddUserPage() {
  const [adminUsername, setAdminUsername] = useState("");
  const [adminPassword, setAdminPassword] = useState("");

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState("Doctor");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setMessage(null);

    try {
      if (password !== confirmPassword) {
        throw new Error("Passwords do not match");
      }
      if (password.length < 8) {
        throw new Error("Password must be at least 8 characters");
      }

      const response = await registerUserWithAdminCredentials({
        admin_username: adminUsername.trim(),
        admin_password: adminPassword,
        username: username.trim().toLowerCase(),
        password,
        role,
      });

      const createdUser = response?.created?.username ?? username.trim().toLowerCase();
      setMessage(`User ${createdUser} created successfully as ${role}.`);
      setUsername("");
      setPassword("");
      setConfirmPassword("");
      setRole("Doctor");
    } catch (exc: unknown) {
      const detail = exc instanceof Error ? exc.message : "Unable to create account";
      setError(detail);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto flex min-h-[78vh] max-w-2xl items-center px-4 sm:px-0">
      <Card className="w-full border-border/60 bg-card/95 p-6 shadow-xl sm:p-7">
        <p role="status" aria-live="polite" className="sr-only">{error || message || ""}</p>
        <div className="mb-5 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <UserRoundCog className="h-5 w-5 text-primary" />
            <h1 className="text-lg font-semibold">Add User / Doctor</h1>
          </div>
          <Link
            href="/login"
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Sign In
          </Link>
        </div>

        <p className="mb-4 text-sm text-muted-foreground">
          This page requires admin credentials to create a new account.
        </p>

        <form className="space-y-4" onSubmit={handleSubmit} aria-busy={submitting}>
          <div className="rounded-lg border border-border/60 bg-muted/30 p-4">
            <h2 className="mb-3 text-sm font-semibold">Admin Verification</h2>

            <div className="mb-3">
              <Label className="mb-1.5 text-xs text-muted-foreground">Admin Username</Label>
              <Input
                value={adminUsername}
                onChange={(event) => setAdminUsername(event.target.value)}
                autoComplete="username"
                required
              />
            </div>

            <div>
              <Label className="mb-1.5 text-xs text-muted-foreground">Admin Password</Label>
              <Input
                type="password"
                value={adminPassword}
                onChange={(event) => setAdminPassword(event.target.value)}
                autoComplete="current-password"
                required
              />
            </div>
          </div>

          <div className="rounded-lg border border-border/60 p-4">
            <h2 className="mb-3 text-sm font-semibold">New Account</h2>

            <div className="mb-3">
              <Label className="mb-1.5 text-xs text-muted-foreground">Username</Label>
              <Input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                autoComplete="off"
                placeholder="doctor_jane"
                required
              />
            </div>

            <div className="mb-3">
              <Label className="mb-1.5 text-xs text-muted-foreground">Temporary Password</Label>
              <Input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="new-password"
                required
              />
            </div>

            <div className="mb-3">
              <Label className="mb-1.5 text-xs text-muted-foreground">Confirm Password</Label>
              <Input
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                autoComplete="new-password"
                required
              />
            </div>

            <div>
              <Label className="mb-1.5 text-xs text-muted-foreground">Role</Label>
              <select
                className="h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm sm:h-9"
                value={role}
                onChange={(event) => setRole(event.target.value)}
                aria-label="New account role"
              >
                <option value="Doctor">Doctor</option>
                <option value="Nurse">Nurse</option>
                <option value="Admin">Admin</option>
              </select>
            </div>
          </div>

          {error ? <p className="text-sm text-destructive" role="alert">{error}</p> : null}
          {message ? <p className="text-sm text-emerald-500" role="status" aria-live="polite">{message}</p> : null}

          <Button type="submit" className="min-h-10 w-full px-3 sm:min-h-9" disabled={submitting}>
            {submitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <UserPlus className="mr-2 h-4 w-4" />}
            Create Account
          </Button>
        </form>
      </Card>
    </div>
  );
}
