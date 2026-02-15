"use client";

import { ReactNode, useCallback, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Loader2, Mic, MicOff } from "lucide-react";
import { Sidebar } from "@/components/sidebar";
import { getAuthToken } from "@/lib/api";
import { LanguageMenuButton } from "@/components/language-menu-button";
import { ThemeToggleButton } from "@/components/theme-toggle-button";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/components/language-provider";
import { useVoiceAssistant } from "@/components/voice-assistant-context";
import { VoiceAssistantOverlay } from "@/components/voice-assistant-overlay";

type AppShellProps = {
  children: ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [hasToken, setHasToken] = useState(false);
  const { t } = useI18n();
  const voice = useVoiceAssistant();

  const isLoginRoute = pathname === "/login";
  const isQueueRoute = pathname === "/queue";

  const handleMicClick = useCallback(() => {
    if (voice.state.active) {
      voice.stop();
    } else {
      // Navigate to analysis page first if not already there
      if (pathname !== "/analysis") {
        router.push("/analysis");
        // Small delay so next page mounts and registers form handlers
        setTimeout(() => voice.start(), 800);
      } else {
        voice.start();
      }
    }
  }, [voice, pathname, router]);

  useEffect(() => {
    setMounted(true);
    setHasToken(Boolean(getAuthToken()));
  }, []);

  useEffect(() => {
    if (mounted && !isLoginRoute && !hasToken) {
      router.replace("/login");
    }
  }, [mounted, isLoginRoute, hasToken, router]);

  if (!isLoginRoute && (!mounted || !hasToken)) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-primary" />
      </div>
    );
  }

  if (!mounted) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-primary" />
      </div>
    );
  }

  if (isLoginRoute) {
    return (
      <main className="flex-1">
        <div className="fixed right-4 top-4 z-50 flex items-center gap-2 rtl:right-auto rtl:left-4">
          <ThemeToggleButton />
          <Button
            type="button"
            variant="destructive"
            size="icon"
            aria-label={t("common.microphone")}
            title={voice.state.active ? "Stop Voice Assistant" : "Start Voice Assistant"}
            onClick={handleMicClick}
            className={`size-16 rounded-full text-white shadow-lg transition-all ${
              voice.state.active
                ? "bg-red-600 hover:bg-red-700 animate-pulse ring-4 ring-red-400/30"
                : "bg-gradient-to-br from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700"
            }`}
          >
            {voice.state.active ? <MicOff className="h-8 w-8" /> : <Mic className="h-8 w-8" />}
          </Button>
          <LanguageMenuButton />
        </div>
        <VoiceAssistantOverlay />
        <div className="mx-auto max-w-[1440px] px-6 py-6">{children}</div>
      </main>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="ml-[240px] flex-1 transition-all duration-300 rtl:ml-0 rtl:mr-[240px]">
        <div className="fixed right-4 top-4 z-50 flex items-center gap-2 rtl:right-auto rtl:left-4">
          {!isQueueRoute && <ThemeToggleButton />}
          {!isQueueRoute && (
            <Button
              type="button"
              variant="destructive"
              size="icon"
              aria-label={t("common.microphone")}
              title={voice.state.active ? "Stop Voice Assistant" : "Start Voice Assistant"}
              onClick={handleMicClick}
              className={`size-16 rounded-full text-white shadow-lg transition-all ${
                voice.state.active
                  ? "bg-red-600 hover:bg-red-700 animate-pulse ring-4 ring-red-400/30"
                  : "bg-gradient-to-br from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700"
              }`}
            >
              {voice.state.active ? <MicOff className="h-8 w-8" /> : <Mic className="h-8 w-8" />}
            </Button>
          )}
          <LanguageMenuButton />
        </div>
        <VoiceAssistantOverlay />
        <div className="mx-auto max-w-[1440px] px-6 py-6">{children}</div>
      </main>
    </div>
  );
}
