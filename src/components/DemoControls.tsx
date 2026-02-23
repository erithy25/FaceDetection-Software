import { useState, useEffect, useRef } from "react";

interface DemoControlsProps {
  mode: "live" | "deepfake";
  onSwitchMode: (mode: "live" | "deepfake") => void;
}

/**
 * Demo mode controls for competition presentations. Provides a
 * one-click toggle between live webcam and pre-recorded deepfake
 * input, with an injection timer showing detection speed.
 */
export default function DemoControls({ mode, onSwitchMode }: DemoControlsProps) {
  const [injectionTime, setInjectionTime] = useState<number | null>(null);
  const switchTimestamp = useRef<number | null>(null);

  useEffect(() => {
    if (mode === "deepfake") {
      switchTimestamp.current = Date.now();
      setInjectionTime(null);
    } else {
      switchTimestamp.current = null;
      setInjectionTime(null);
    }
  }, [mode]);

  const handleSwitch = () => {
    const newMode = mode === "live" ? "deepfake" : "live";
    onSwitchMode(newMode);
  };

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

      {injectionTime !== null && (
        <span className="font-mono text-xs text-accent-yellow">
          Detected in {injectionTime}ms
        </span>
      )}
    </div>
  );
}
