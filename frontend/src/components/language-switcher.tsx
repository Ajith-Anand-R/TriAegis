"use client";

import { SUPPORTED_LANGUAGES } from "@/lib/i18n";
import { useI18n } from "@/components/language-provider";
import { LanguageCode } from "@/lib/i18n";

type LanguageSwitcherProps = {
  compact?: boolean;
};

export function LanguageSwitcher({ compact = false }: LanguageSwitcherProps) {
  const { language, setLanguage, t } = useI18n();
  const label = t("lang.label");

  return (
    <div className={compact ? "" : "space-y-1"}>
      {!compact ? <p className="text-xs text-muted-foreground">{label}</p> : null}
      <select
        aria-label={label}
        title={label}
        value={language}
        onChange={(event) => setLanguage(event.target.value as LanguageCode)}
        className="flex h-9 w-full rounded-md border border-input bg-input/50 px-3 py-1 text-xs shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        {SUPPORTED_LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.label}
          </option>
        ))}
      </select>
    </div>
  );
}
