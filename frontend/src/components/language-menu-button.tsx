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

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="bg-background/95 backdrop-blur">
          <Languages className="h-4 w-4" />
          {t("lang.label")} • {current?.label ?? "English"}
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
