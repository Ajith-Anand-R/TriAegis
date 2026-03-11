"use client";

import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from "react";
import { DEFAULT_LANGUAGE, LanguageCode, TRANSLATIONS, TranslationKey } from "@/lib/i18n";

type I18nContextType = {
  language: LanguageCode;
  setLanguage: (language: LanguageCode) => void;
  t: (key: TranslationKey) => string;
};

const I18nContext = createContext<I18nContextType | null>(null);

const STORAGE_KEY = "triaegis.language";
const RTL_LANGUAGES: LanguageCode[] = ["ar"];

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<LanguageCode>(() => {
    if (typeof window === "undefined") {
      return DEFAULT_LANGUAGE;
    }

    const stored = window.localStorage.getItem(STORAGE_KEY) as LanguageCode | null;
    if (stored && TRANSLATIONS[stored]) {
      return stored;
    }

    return DEFAULT_LANGUAGE;
  });

  const value = useMemo<I18nContextType>(
    () => {
      const languageTranslations = TRANSLATIONS[language] as Partial<
        Record<TranslationKey, string>
      >;

      return {
        language,
        setLanguage: (nextLanguage: LanguageCode) => {
          setLanguage(nextLanguage);
          window.localStorage.setItem(STORAGE_KEY, nextLanguage);
        },
        t: (key: TranslationKey) => languageTranslations[key] ?? TRANSLATIONS.en[key],
      };
    },
    [language]
  );

  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dir = RTL_LANGUAGES.includes(language) ? "rtl" : "ltr";
  }, [language]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within LanguageProvider");
  }
  return context;
}
