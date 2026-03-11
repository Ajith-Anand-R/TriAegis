"use client";

import { useEffect, useMemo, useRef } from "react";
import { X, Mic, Volume2, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { useVoiceAssistant, VOICE_STEPS_META, VoiceStep } from "./voice-assistant-context";

/* ------------------------------------------------------------------ */
/*  Animated Orb — CSS-only pulse / waveform                           */
/* ------------------------------------------------------------------ */
function VoiceOrb({ mode }: { mode: "listening" | "speaking" | "processing" | "idle" }) {
  return (
    <div className="relative flex items-center justify-center">
      {/* Outer rings */}
      {(mode === "listening" || mode === "speaking") && (
        <>
          <span
            className={`absolute h-28 w-28 rounded-full opacity-10 ${
              mode === "listening"
                ? "animate-ping bg-red-500"
                : "animate-pulse bg-blue-500"
            }`}
            style={{ animationDuration: mode === "listening" ? "1.5s" : "2s" }}
          />
          <span
            className={`absolute h-20 w-20 rounded-full opacity-20 ${
              mode === "listening"
                ? "animate-ping bg-red-400"
                : "animate-pulse bg-blue-400"
            }`}
            style={{ animationDuration: mode === "listening" ? "1.8s" : "2.5s" }}
          />
        </>
      )}
      {mode === "processing" && (
        <span className="absolute h-20 w-20 rounded-full bg-amber-500/20 animate-pulse" />
      )}

      {/* Core orb */}
      <div
        className={`relative z-10 flex h-16 w-16 items-center justify-center rounded-full shadow-lg transition-all duration-300 ${
          mode === "listening"
            ? "bg-gradient-to-br from-red-500 to-rose-600 shadow-red-500/40 scale-110"
            : mode === "speaking"
            ? "bg-gradient-to-br from-blue-500 to-indigo-600 shadow-blue-500/40 scale-105"
            : mode === "processing"
            ? "bg-gradient-to-br from-amber-500 to-orange-600 shadow-amber-500/40"
            : "bg-gradient-to-br from-slate-500 to-slate-600 shadow-slate-500/20"
        }`}
      >
        {mode === "listening" && <Mic className="h-7 w-7 text-white animate-pulse" />}
        {mode === "speaking" && <Volume2 className="h-7 w-7 text-white animate-pulse" />}
        {mode === "processing" && <Loader2 className="h-7 w-7 text-white animate-spin" />}
        {mode === "idle" && <Mic className="h-7 w-7 text-white/70" />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Soundwave Bars                                                     */
/* ------------------------------------------------------------------ */
function SoundwaveBars({ active, color }: { active: boolean; color: string }) {
  const bars = useMemo(
    () =>
      [
        { height: 10, duration: 0.34, delay: 0.0 },
        { height: 16, duration: 0.42, delay: 0.05 },
        { height: 13, duration: 0.36, delay: 0.1 },
        { height: 20, duration: 0.48, delay: 0.15 },
        { height: 12, duration: 0.33, delay: 0.2 },
        { height: 18, duration: 0.44, delay: 0.25 },
        { height: 14, duration: 0.38, delay: 0.3 },
        { height: 22, duration: 0.5, delay: 0.35 },
        { height: 11, duration: 0.35, delay: 0.4 },
        { height: 17, duration: 0.43, delay: 0.45 },
        { height: 15, duration: 0.39, delay: 0.5 },
        { height: 19, duration: 0.46, delay: 0.55 },
      ],
    []
  );

  return (
    <div className="flex items-end gap-0.5 h-6">
      {bars.map((bar, i) => (
        <div
          key={i}
          className={`w-1 rounded-full transition-all ${color}`}
          style={{
            height: active ? `${bar.height}px` : "4px",
            opacity: active ? 0.8 : 0.2,
            animationName: active ? "voicebar" : "none",
            animationDuration: active ? `${bar.duration}s` : "0s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
            animationDirection: "alternate",
            animationDelay: `${bar.delay}s`,
          }}
        />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Progress Steps                                                     */
/* ------------------------------------------------------------------ */
function getStepIndex(step: VoiceStep): number {
  return VOICE_STEPS_META.findIndex((m) => m.step === step);
}

function StepProgress({ currentStep }: { currentStep: VoiceStep }) {
  const currentIdx = getStepIndex(currentStep);
  const isDone = currentStep === "complete";

  return (
    <div className="flex items-center gap-1 overflow-x-auto py-1">
      {VOICE_STEPS_META.map((meta, i) => {
        const isActive = meta.step === currentStep;
        const isCompleted = isDone || (currentIdx > i);
        return (
          <div key={meta.step} className="flex items-center gap-1">
            <div
              className={`flex h-6 min-w-6 items-center justify-center rounded-full text-[10px] font-bold transition-all ${
                isActive
                  ? "bg-primary text-primary-foreground scale-110 shadow-md"
                  : isCompleted
                  ? "bg-emerald-500 text-white"
                  : "bg-muted text-muted-foreground"
              }`}
              title={meta.label}
            >
              {isCompleted && !isActive ? (
                <CheckCircle className="h-3.5 w-3.5" />
              ) : (
                <span>{meta.icon}</span>
              )}
            </div>
            {i < VOICE_STEPS_META.length - 1 && (
              <div
                className={`h-0.5 w-3 rounded-full transition-all ${
                  isCompleted ? "bg-emerald-500" : "bg-muted"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Countdown Ring                                                     */
/* ------------------------------------------------------------------ */
function CountdownRing({ seconds }: { seconds: number }) {
  const pct = (seconds / 5) * 100;
  const radius = 36;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (pct / 100) * circ;

  return (
    <div className="relative flex items-center justify-center">
      <svg width="88" height="88" className="-rotate-90">
        <circle cx="44" cy="44" r={radius} fill="none" stroke="currentColor" strokeWidth="4" className="text-muted/30" />
        <circle
          cx="44"
          cy="44"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth="4"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="text-primary transition-all duration-1000"
        />
      </svg>
      <span className="absolute text-2xl font-bold tabular-nums">{seconds}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Overlay                                                       */
/* ------------------------------------------------------------------ */
export function VoiceAssistantOverlay() {
  const { state, stop } = useVoiceAssistant();
  const panelRef = useRef<HTMLDivElement>(null);

  // Prevent scroll behind overlay
  useEffect(() => {
    if (state.active) {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [state.active]);

  if (!state.active && state.step === "idle") return null;

  const { step, subState, prompt, transcript, error, countdown, resultText } = state;

  const statusLabel =
    subState === "listening"
      ? "Listening..."
      : subState === "speaking"
      ? "Speaking..."
      : subState === "processing"
      ? "Processing..."
      : step === "complete"
      ? "Session Complete"
      : "Initializing...";

  const statusColor =
    subState === "listening"
      ? "text-red-400"
      : subState === "speaking"
      ? "text-blue-400"
      : subState === "processing"
      ? "text-amber-400"
      : "text-muted-foreground";

  return (
    <>
      {/* Keyframes for soundwave */}
      <style>{`
        @keyframes voicebar {
          0% { transform: scaleY(0.4); }
          100% { transform: scaleY(1); }
        }
      `}</style>

      {/* Backdrop */}
      <div className="fixed inset-0 z-[100] bg-black/40 backdrop-blur-sm" onClick={stop} />

      {/* Panel */}
      <div
        ref={panelRef}
        className="fixed z-[101] bottom-6 right-6 w-[380px] max-w-[calc(100vw-2rem)] rounded-2xl border border-border/60 bg-card shadow-2xl animate-in slide-in-from-bottom-4 duration-300"
        style={{ maxHeight: "calc(100vh - 3rem)" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/40 px-5 py-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
              <Mic className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">TriAegis Voice Assistant</h3>
              <p className={`text-[11px] font-medium ${statusColor}`}>{statusLabel}</p>
            </div>
          </div>
          <button
            onClick={stop}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-muted/50 text-muted-foreground transition-colors hover:bg-destructive hover:text-destructive-foreground"
            title="Stop & Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="overflow-y-auto px-5 py-4 space-y-4" style={{ maxHeight: "440px" }}>
          {/* Step Progress */}
          <StepProgress currentStep={step} />

          {/* Orb + Soundwave */}
          <div className="flex flex-col items-center gap-3 py-2">
            {step === "countdown" ? (
              <CountdownRing seconds={countdown} />
            ) : (
              <VoiceOrb mode={subState} />
            )}
            <SoundwaveBars
              active={subState === "listening" || subState === "speaking"}
              color={subState === "listening" ? "bg-red-400" : "bg-blue-400"}
            />
          </div>

          {/* Current prompt */}
          {prompt && (
            <div className="rounded-xl bg-primary/5 border border-primary/10 px-4 py-3">
              <p className="text-sm leading-relaxed text-foreground/90">{prompt}</p>
            </div>
          )}

          {/* Transcript */}
          {transcript && (
            <div className="rounded-xl bg-muted/40 border border-border/40 px-4 py-3">
              <p className="text-[11px] font-medium text-muted-foreground mb-1">You said:</p>
              <p className="text-sm text-foreground/80 italic">&ldquo;{transcript}&rdquo;</p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 rounded-xl bg-destructive/10 border border-destructive/20 px-4 py-3">
              <AlertCircle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
              <p className="text-xs text-destructive">{error}</p>
            </div>
          )}

          {/* Result transcript */}
          {resultText && step === "result" && (
            <div className="rounded-xl bg-emerald-500/5 border border-emerald-500/20 px-4 py-3">
              <p className="text-[11px] font-medium text-emerald-500 mb-1">Result Summary</p>
              <p className="text-xs text-foreground/80 leading-relaxed">{resultText}</p>
            </div>
          )}

          {/* Complete state */}
          {step === "complete" && (
            <div className="flex flex-col items-center gap-2 py-2">
              <CheckCircle className="h-10 w-10 text-emerald-500" />
              <p className="text-sm font-medium text-emerald-500">Session Complete</p>
              <p className="text-xs text-muted-foreground text-center">
                You can review the results on screen or start a new session.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border/40 px-5 py-3 flex items-center justify-between">
          <p className="text-[10px] text-muted-foreground">
            Powered by Web Speech API
          </p>
          <button
            onClick={stop}
            className="rounded-lg bg-destructive/10 px-3 py-1.5 text-xs font-medium text-destructive transition-colors hover:bg-destructive/20"
          >
            {step === "complete" ? "Close" : "Stop Assistant"}
          </button>
        </div>
      </div>
    </>
  );
}
