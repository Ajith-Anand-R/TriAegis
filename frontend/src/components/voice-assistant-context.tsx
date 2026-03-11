"use client";

import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export type VoiceStep =
  | "idle"
  | "greeting"
  | "age"
  | "gender"
  | "symptoms"
  | "conditions"
  | "ehr"
  | "countdown"
  | "analyzing"
  | "result"
  | "complete";

export type VoiceSubState = "speaking" | "listening" | "processing" | "idle";

export interface VoiceStepMeta {
  step: VoiceStep;
  label: string;
  icon: string;
}

export const VOICE_STEPS_META: VoiceStepMeta[] = [
  { step: "greeting", label: "Greeting", icon: "👋" },
  { step: "age", label: "Age", icon: "🎂" },
  { step: "gender", label: "Gender", icon: "⚧" },
  { step: "symptoms", label: "Symptoms", icon: "🩺" },
  { step: "conditions", label: "Conditions", icon: "🛡️" },
  { step: "ehr", label: "EHR Upload", icon: "📄" },
  { step: "countdown", label: "Analyzing", icon: "⏳" },
  { step: "result", label: "Results", icon: "📊" },
];

/** Handlers the Analysis page registers so the voice bot can fill form fields */
export interface AnalysisFormHandlers {
  setAge: (v: number) => void;
  setGender: (v: string) => void;
  setSelectedSymptoms: (v: string[]) => void;
  setSelectedConditions: (v: string[]) => void;
  triggerFileUpload: () => void;
  triggerAnalyze: () => Promise<void>;
  getSymptomOptions: () => string[];
  getConditionOptions: () => string[];
  getResult: () => unknown;
}

export interface VoiceAssistantState {
  active: boolean;
  step: VoiceStep;
  subState: VoiceSubState;
  transcript: string;
  prompt: string;
  error: string | null;
  countdown: number;
  resultText: string;
}

export interface VoiceAssistantAPI {
  state: VoiceAssistantState;
  start: () => void;
  stop: () => void;
  registerFormHandlers: (handlers: AnalysisFormHandlers) => void;
  unregisterFormHandlers: () => void;
}

const VoiceAssistantContext = createContext<VoiceAssistantAPI | null>(null);

/* ------------------------------------------------------------------ */
/*  Speech helpers                                                     */
/* ------------------------------------------------------------------ */

type BrowserSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  continuous: boolean;
  onresult: ((event: { results?: ArrayLike<ArrayLike<{ transcript?: string }>> }) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  onspeechend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type BrowserSpeechRecognitionCtor = new () => BrowserSpeechRecognition;

type SpeechWindow = Window & {
  SpeechRecognition?: BrowserSpeechRecognitionCtor;
  webkitSpeechRecognition?: BrowserSpeechRecognitionCtor;
};

function getSpeechRecognitionCtor(): BrowserSpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as SpeechWindow;
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

function matchOptions(rawText: string, options: string[]): string[] {
  const cleaned = rawText.trim().toLowerCase().replace(/\band\b/g, ",");
  const chunks = cleaned
    .split(/[,.;/]+/)
    .map((c) => c.trim())
    .filter(Boolean);

  return options.filter((opt) => {
    const norm = opt.toLowerCase();
    return chunks.some((c) => c.includes(norm) || norm.includes(c));
  });
}

/* ------------------------------------------------------------------ */
/*  Provider                                                           */
/* ------------------------------------------------------------------ */

export function VoiceAssistantProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<VoiceAssistantState>({
    active: false,
    step: "idle",
    subState: "idle",
    transcript: "",
    prompt: "",
    error: null,
    countdown: 0,
    resultText: "",
  });

  const formRef = useRef<AnalysisFormHandlers | null>(null);
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortedRef = useRef(false);
  const activeRef = useRef(false);
  const stepRef = useRef<VoiceStep>("idle");

  // Keep refs in sync with state for use in callbacks
  useEffect(() => {
    activeRef.current = state.active;
    stepRef.current = state.step;
  }, [state.active, state.step]);

  /* ---------- TTS ---------- */
  const speak = useCallback(
    (text: string): Promise<void> =>
      new Promise((resolve) => {
        if (typeof window === "undefined" || !("speechSynthesis" in window)) {
          resolve();
          return;
        }
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "en-US";
        utterance.rate = 1.05;
        utterance.pitch = 1.0;

        // Try to pick a nicer voice
        const voices = window.speechSynthesis.getVoices();
        const preferred = voices.find(
          (v) =>
            v.lang.startsWith("en") &&
            (v.name.includes("Google") ||
              v.name.includes("Samantha") ||
              v.name.includes("Microsoft") ||
              v.name.includes("Natural"))
        );
        if (preferred) utterance.voice = preferred;

        utterance.onend = () => resolve();
        utterance.onerror = () => resolve();
        window.speechSynthesis.speak(utterance);
      }),
    []
  );

  /* ---------- STT ---------- */
  const listen = useCallback((): Promise<string> => {
    return new Promise((resolve, reject) => {
      const Ctor = getSpeechRecognitionCtor();
      if (!Ctor) {
        reject(new Error("Speech recognition not supported"));
        return;
      }

      const rec = new Ctor();
      rec.lang = "en-US";
      rec.interimResults = false;
      rec.maxAlternatives = 1;
      rec.continuous = false;
      recognitionRef.current = rec;

      let resolved = false;

      rec.onresult = (e) => {
        const transcript = e.results?.[0]?.[0]?.transcript ?? "";
        resolved = true;
        resolve(transcript);
      };

      rec.onerror = (e) => {
        if (!resolved) {
          resolved = true;
          reject(new Error(e.error || "recognition error"));
        }
      };

      rec.onend = () => {
        recognitionRef.current = null;
        if (!resolved) {
          resolved = true;
          resolve(""); // silence / no speech detected
        }
      };

      rec.start();
    });
  }, []);

  /* ---------- step update helper ---------- */
  const patch = useCallback(
    (partial: Partial<VoiceAssistantState>) =>
      setState((prev) => ({ ...prev, ...partial })),
    []
  );

  /* ---------- cleanup ---------- */
  const cleanup = useCallback(() => {
    abortedRef.current = true;
    recognitionRef.current?.abort();
    recognitionRef.current = null;
    if (countdownRef.current) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  /* ---------- run step ---------- */
  const runStep = useCallback(
    async (step: VoiceStep, retryCount = 0): Promise<void> => {
      if (abortedRef.current) return;

      const form = formRef.current;
      if (!form && step !== "greeting" && step !== "idle" && step !== "complete") {
        patch({ error: "Analysis page not ready. Please navigate to Analysis first." });
        return;
      }

      patch({ step, error: null, transcript: "" });

      const MAX_RETRIES = 2;

      const askAndListen = async (
        prompt: string,
        nextStep: VoiceStep,
        handler: (text: string) => boolean // returns true if valid
      ): Promise<void> => {
        if (abortedRef.current) return;

        patch({ subState: "speaking", prompt });
        await speak(prompt);
        if (abortedRef.current) return;

        patch({ subState: "listening" });
        try {
          const text = await listen();
          if (abortedRef.current) return;

          patch({ transcript: text, subState: "processing" });

          if (handler(text)) {
            await runStep(nextStep);
          } else {
            if (retryCount < MAX_RETRIES) {
              patch({ error: "I didn't quite catch that. Let me ask again." });
              await new Promise((r) => setTimeout(r, 500));
              await runStep(step, retryCount + 1);
            } else {
              patch({
                error: "Could not understand after multiple attempts. Moving to next step.",
              });
              await new Promise((r) => setTimeout(r, 1000));
              await runStep(nextStep);
            }
          }
        } catch {
          if (abortedRef.current) return;
          if (retryCount < MAX_RETRIES) {
            patch({ error: "Microphone error. Retrying..." });
            await new Promise((r) => setTimeout(r, 800));
            await runStep(step, retryCount + 1);
          } else {
            patch({ error: "Microphone issue. Skipping to next step." });
            await new Promise((r) => setTimeout(r, 1000));
            await runStep(nextStep);
          }
        }
      };

      switch (step) {
        case "greeting": {
          const greeting = `${getGreeting()}! I'm your TriAegis voice assistant. I'll help you complete the patient analysis. Let's get started.`;
          patch({ subState: "speaking", prompt: greeting });
          await speak(greeting);
          if (abortedRef.current) return;
          await runStep("age");
          break;
        }

        case "age": {
          await askAndListen(
            "What is the patient's age?",
            "gender",
            (text) => {
              const match = text.match(/\b(\d{1,3})\b/);
              const parsed = match ? Number(match[1]) : NaN;
              if (!Number.isNaN(parsed) && parsed >= 0 && parsed <= 120) {
                form!.setAge(parsed);
                return true;
              }
              return false;
            }
          );
          break;
        }

        case "gender": {
          await askAndListen(
            "What is the patient's gender? Male, female, or other?",
            "symptoms",
            (text) => {
              const lower = text.toLowerCase();
              let g = "";
              if (lower.includes("female") || lower.includes("woman") || lower.includes("girl")) g = "Female";
              else if (lower.includes("male") || lower.includes("man") || lower.includes("boy")) g = "Male";
              else if (lower.includes("other") || lower.includes("non")) g = "Other";
              if (g) {
                form!.setGender(g);
                return true;
              }
              return false;
            }
          );
          break;
        }

        case "symptoms": {
          const options = form!.getSymptomOptions();
          const optionsList = options.length > 8
            ? options.slice(0, 8).join(", ") + ", and more"
            : options.join(", ");
          await askAndListen(
            `Please tell me the patient's symptoms. Available options include: ${optionsList}. You can say multiple symptoms separated by 'and'.`,
            "conditions",
            (text) => {
              const lower = text.toLowerCase();
              if (lower.includes("none") || lower.includes("no symptom") || lower.includes("nothing")) {
                form!.setSelectedSymptoms([]);
                return true;
              }
              const matched = matchOptions(text, options);
              if (matched.length > 0) {
                form!.setSelectedSymptoms(matched);
                return true;
              }
              return false;
            }
          );
          break;
        }

        case "conditions": {
          const options = form!.getConditionOptions();
          const optionsList = options.length > 8
            ? options.slice(0, 8).join(", ") + ", and more"
            : options.join(", ");
          await askAndListen(
            `Now, does the patient have any pre-existing conditions? Options include: ${optionsList}. Say none if there aren't any.`,
            "ehr",
            (text) => {
              const lower = text.toLowerCase();
              if (lower.includes("none") || lower.includes("no condition") || lower.includes("nothing") || lower.includes("no ")) {
                form!.setSelectedConditions([]);
                return true;
              }
              const matched = matchOptions(text, options);
              if (matched.length > 0) {
                form!.setSelectedConditions(matched);
                return true;
              }
              return false;
            }
          );
          break;
        }

        case "ehr": {
          await askAndListen(
            "Would you like to upload an EMR or EHR document? Say yes to open the file picker, or say no to skip.",
            "countdown",
            (text) => {
              const lower = text.toLowerCase();
              if (/(yes|yeah|ready|upload|sure|okay|ok)/.test(lower)) {
                form!.triggerFileUpload();
              }
              // Always proceed regardless
              return true;
            }
          );
          break;
        }

        case "countdown": {
          const countdownPrompt = "Thank you! All information collected. Starting analysis in 5 seconds.";
          patch({ subState: "speaking", prompt: countdownPrompt, countdown: 5 });
          await speak(countdownPrompt);
          if (abortedRef.current) return;

          // Countdown timer
          await new Promise<void>((resolve) => {
            let remaining = 5;
            patch({ countdown: remaining, subState: "processing" });
            countdownRef.current = setInterval(() => {
              remaining--;
              patch({ countdown: remaining });
              if (remaining <= 0) {
                if (countdownRef.current) {
                  clearInterval(countdownRef.current);
                  countdownRef.current = null;
                }
                resolve();
              }
            }, 1000);
          });

          if (abortedRef.current) return;
          await runStep("analyzing");
          break;
        }

        case "analyzing": {
          patch({ subState: "processing", prompt: "Analyzing patient data..." });
          try {
            await form!.triggerAnalyze();
            if (abortedRef.current) return;
            // Wait a moment for result state to propagate
            await new Promise((r) => setTimeout(r, 800));
            await runStep("result");
          } catch {
            patch({ error: "Analysis failed. Please try again manually." });
          }
          break;
        }

        case "result": {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const result = form!.getResult() as any;
          if (!result?.prediction) {
            patch({ error: "No results available yet.", subState: "idle" });
            await new Promise((r) => setTimeout(r, 2000));
            await runStep("complete");
            return;
          }

          const pred = result.prediction;
          const parts: string[] = [
            `Analysis complete.`,
            `The patient's risk level is ${pred.risk_level}.`,
            `Priority score is ${pred.priority_score} out of 10.`,
            `Confidence is ${(pred.confidence * 100).toFixed(0)} percent.`,
          ];

          if (pred.department) {
            parts.push(`Recommended department: ${pred.department}.`);
          }
          if (pred.estimated_wait_time) {
            parts.push(`Estimated wait time: ${pred.estimated_wait_time}.`);
          }
          if (result.clinical_explanation) {
            parts.push(String(result.clinical_explanation));
          }
          if (result.manual_review_recommended) {
            parts.push("Manual clinical review is recommended for this patient.");
          }

          const resultText = parts.join(" ");
          patch({ subState: "speaking", prompt: "Reading results...", resultText });
          await speak(resultText);
          if (abortedRef.current) return;
          await runStep("complete");
          break;
        }

        case "complete": {
          patch({
            subState: "idle",
            prompt: "Voice assistant session complete. You can review the results on screen.",
          });
          break;
        }

        default:
          break;
      }
    },
    [speak, listen, patch]
  );

  /* ---------- Public API ---------- */
  const start = useCallback(() => {
    abortedRef.current = false;
    patch({
      active: true,
      step: "greeting",
      subState: "idle",
      transcript: "",
      prompt: "",
      error: null,
      countdown: 0,
      resultText: "",
    });
    // Small delay to let navigation happen before voice starts
    setTimeout(() => {
      if (!abortedRef.current) {
        runStep("greeting");
      }
    }, 600);
  }, [patch, runStep]);

  const stop = useCallback(() => {
    cleanup();
    patch({
      active: false,
      step: "idle",
      subState: "idle",
      transcript: "",
      prompt: "",
      error: null,
      countdown: 0,
      resultText: "",
    });
  }, [cleanup, patch]);

  const registerFormHandlers = useCallback((handlers: AnalysisFormHandlers) => {
    formRef.current = handlers;
  }, []);

  const unregisterFormHandlers = useCallback(() => {
    formRef.current = null;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  const api: VoiceAssistantAPI = {
    state,
    start,
    stop,
    registerFormHandlers,
    unregisterFormHandlers,
  };

  return (
    <VoiceAssistantContext.Provider value={api}>
      {children}
    </VoiceAssistantContext.Provider>
  );
}

export function useVoiceAssistant(): VoiceAssistantAPI {
  const ctx = useContext(VoiceAssistantContext);
  if (!ctx) {
    throw new Error("useVoiceAssistant must be used within VoiceAssistantProvider");
  }
  return ctx;
}
