"use client";

import { Languages } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useI18n } from "@/components/language-provider";
import { LanguageCode, SUPPORTED_LANGUAGES } from "@/lib/i18n";

export function LanguageMenuButton() {
  const { language, setLanguage, t } = useI18n();
  const current = SUPPORTED_LANGUAGES.find((lang) => lang.code === language);
  const compactLabel = (current?.code ?? "en").toUpperCase();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="h-10 bg-background/95 px-2.5 backdrop-blur sm:h-9 sm:px-3"
          aria-label={t("lang.label")}
        >
          <Languages className="h-4 w-4" />
          <span className="hidden sm:inline">{t("lang.label")} • {current?.label ?? "English"}</span>
          <span className="text-xs font-semibold sm:hidden">{compactLabel}</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuLabel>{t("lang.label")}</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={language}
          onValueChange={(value) => setLanguage(value as LanguageCode)}
        >
          {SUPPORTED_LANGUAGES.map((lang) => (
            <DropdownMenuRadioItem key={lang.code} value={lang.code}>
              {lang.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
