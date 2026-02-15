"use client";

import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/components/theme-provider";
import { useI18n } from "@/components/language-provider";

export function ThemeToggleButton() {
  const { theme, toggleTheme } = useTheme();
  const { t } = useI18n();
  const nextThemeLabel = theme === "dark" ? t("theme.switchToLight") : t("theme.switchToDark");

  return (
    <Button
      type="button"
      variant="outline"
      size="icon-sm"
      onClick={toggleTheme}
      aria-label={nextThemeLabel}
      title={nextThemeLabel}
      className="bg-background/95 backdrop-blur"
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}
