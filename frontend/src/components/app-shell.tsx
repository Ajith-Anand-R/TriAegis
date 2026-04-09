"use client";

import { ReactNode, useCallback, useEffect, useSyncExternalStore } from "react";
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
  const { t } = useI18n();
  const voice = useVoiceAssistant();
  const isHydrated = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  const isLoginRoute = pathname === "/login" || pathname.startsWith("/login/");
  const isQueueRoute = pathname === "/queue";
  const hasToken = isHydrated ? Boolean(getAuthToken()) : false;

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
    if (!isHydrated) {
      return;
    }

    if (!isLoginRoute && !hasToken) {
      router.replace("/login");
    }
    if (isLoginRoute && hasToken) {
      router.replace("/");
    }
  }, [isHydrated, isLoginRoute, hasToken, router]);

  if (!isHydrated) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-primary" />
      </div>
    );
  }

  if (!isLoginRoute && !hasToken) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-primary" />
      </div>
    );
  }

  if (isLoginRoute) {
    return (
      <main className="flex-1">
        <a
          href="#app-main-content"
          className="sr-only fixed left-2 top-2 z-[90] rounded-md bg-background px-3 py-2 text-sm font-medium text-foreground shadow-lg focus:not-sr-only"
        >
          Skip to main content
        </a>
        <div className="fixed right-3 top-3 z-50 flex items-center gap-2 rtl:right-auto rtl:left-3 sm:right-4 sm:top-4 sm:rtl:left-4">
          <ThemeToggleButton />
          <LanguageMenuButton />
        </div>
        <VoiceAssistantOverlay />
        <div id="app-main-content" tabIndex={-1} className="mx-auto max-w-[1440px] px-4 py-4 sm:px-6 sm:py-6">{children}</div>
      </main>
    );
  }

  return (
    <div className="flex min-h-screen">
      <a
        href="#app-main-content"
        className="sr-only fixed left-2 top-2 z-[90] rounded-md bg-background px-3 py-2 text-sm font-medium text-foreground shadow-lg focus:not-sr-only"
      >
        Skip to main content
      </a>
      <Sidebar />
      <main className="flex-1 transition-all duration-300 lg:ml-[240px] rtl:lg:ml-0 rtl:lg:mr-[240px]">
        <div className="fixed right-3 top-3 z-50 flex items-center gap-2 rtl:right-auto rtl:left-3 sm:right-4 sm:top-4 sm:rtl:left-4">
          {!isQueueRoute && <ThemeToggleButton />}
          {!isQueueRoute && (
            <Button
              type="button"
              variant="destructive"
              size="icon"
              aria-label={t("common.microphone")}
              title={voice.state.active ? "Stop Voice Assistant" : "Start Voice Assistant"}
              onClick={handleMicClick}
              className={`size-12 rounded-full text-white shadow-lg transition-all sm:size-16 ${
                voice.state.active
                  ? "bg-red-600 hover:bg-red-700 animate-pulse ring-4 ring-red-400/30"
                  : "bg-gradient-to-br from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700"
              }`}
            >
              {voice.state.active ? <MicOff className="h-6 w-6 sm:h-8 sm:w-8" /> : <Mic className="h-6 w-6 sm:h-8 sm:w-8" />}
            </Button>
          )}
          <LanguageMenuButton />
        </div>
        <VoiceAssistantOverlay />
        <div id="app-main-content" tabIndex={-1} className="mx-auto max-w-[1440px] px-4 py-4 sm:px-6 sm:py-6">{children}</div>
      </main>
    </div>
  );
}
