import { useState, useEffect, useRef, useCallback } from "react";

interface DemoControlsProps {
  mode: "live" | "deepfake";
  onSwitchMode: (mode: "live" | "deepfake") => void;
  currentStatus?: "authentic" | "suspicious" | "deepfake" | "standby" | "warming_up";
}

/**
 * Demo mode controls for competition presentations. Provides a
 * one-click toggle between live webcam and pre-recorded deepfake
 * input, with an injection timer measuring detection speed (ms
 * between deepfake injection and first red alert).
 */
export default function DemoControls({
  mode,
  onSwitchMode,
  currentStatus,
}: DemoControlsProps) {
  const [injectionTime, setInjectionTime] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const switchTimestamp = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  // Start timer when switching to deepfake mode
  useEffect(() => {
    if (mode === "deepfake") {
      switchTimestamp.current = Date.now();
      setInjectionTime(null);
      setElapsedMs(0);
      timerRef.current = setInterval(() => {
        if (switchTimestamp.current) {
          setElapsedMs(Date.now() - switchTimestamp.current);
        }
      }, 10);
    } else {
      switchTimestamp.current = null;
      setElapsedMs(null);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [mode]);

  // Detect when the status changes to "deepfake" after injection
  const detectionRecorded = useRef(false);
  useEffect(() => {
    if (mode === "deepfake" && !detectionRecorded.current) {
      detectionRecorded.current = false;
    }
    if (mode === "live") {
      detectionRecorded.current = false;
    }
  }, [mode]);

  useEffect(() => {
    if (
      mode === "deepfake" &&
      currentStatus === "deepfake" &&
      switchTimestamp.current &&
      !detectionRecorded.current
    ) {
      const elapsed = Date.now() - switchTimestamp.current;
      setInjectionTime(elapsed);
      detectionRecorded.current = true;
      if (timerRef.current) clearInterval(timerRef.current);
    }
  }, [currentStatus, mode]);

  const handleSwitch = useCallback(() => {
    const newMode = mode === "live" ? "deepfake" : "live";
    onSwitchMode(newMode);
  }, [mode, onSwitchMode]);

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={handleSwitch}
        className={`rounded-button px-3 py-1.5 text-sm font-medium transition-colors duration-300 ${
          mode === "deepfake"
            ? "bg-accent-red/20 text-accent-red hover:bg-accent-red/30"
            : "border border-border bg-surface text-text-secondary hover:text-text-primary"
        }`}
      >
        {mode === "deepfake" ? "Stop Deepfake" : "Inject Deepfake"}
      </button>

      {mode === "deepfake" && injectionTime === null && elapsedMs !== null && (
        <span className="font-mono text-xs text-text-secondary animate-pulse">
          {(elapsedMs / 1000).toFixed(1)}s ...
        </span>
      )}

      {injectionTime !== null && (
        <span className="font-mono text-xs text-accent-yellow">
          Detected in {injectionTime}ms
        </span>
      )}
    </div>
  );
}
