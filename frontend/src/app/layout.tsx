import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/app-shell";
import { TooltipProvider } from "@/components/ui/tooltip";
import { LanguageProvider } from "@/components/language-provider";
import { ThemeProvider } from "@/components/theme-provider";
import { VoiceAssistantProvider } from "@/components/voice-assistant-context";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "TriAegis — AI Patient Triage System",
  description:
    "AI-powered smart patient triage, risk classification, department recommendation, and priority queueing.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(() => {
  try {
    const key = "triaegis.theme";
    const stored = localStorage.getItem(key);
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = stored === "light" || stored === "dark" ? stored : (prefersDark ? "dark" : "light");
    document.documentElement.classList.toggle("dark", theme === "dark");
  } catch (_) {}
})();`,
          }}
        />
      </head>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider>
          <LanguageProvider>
            <VoiceAssistantProvider>
              <TooltipProvider>
                <AppShell>{children}</AppShell>
              </TooltipProvider>
            </VoiceAssistantProvider>
          </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
