import { useState, useCallback, useRef, useEffect } from "react";

interface ScoreHistoryEntry {
  time: string;
  score: number;
}

interface DetectionState {
  mode: "live" | "deepfake";
  gradcamEnabled: boolean;
  scoreHistory: ScoreHistoryEntry[];
  switchMode: (mode: "live" | "deepfake") => void;
  toggleGradcam: (enabled: boolean) => void;
}

const MAX_HISTORY_POINTS = 120; // 60 seconds at ~2 points/sec

/**
 * State management hook for detection settings and score history.
 * Sends control commands to the Python backend via WebSocket.
 */
export function useDetectionState(): DetectionState {
  const [mode, setMode] = useState<"live" | "deepfake">("live");
  const [gradcamEnabled, setGradcamEnabled] = useState(false);
  const [scoreHistory, setScoreHistory] = useState<ScoreHistoryEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    try {
      const ws = new WebSocket("ws://localhost:9734");
      wsRef.current = ws;
      return () => ws.close();
    } catch {
      // Connection handled by useWebSocket hook
    }
  }, []);

  const sendCommand = useCallback(
    (command: string, params: Record<string, unknown> = {}) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ command, ...params }));
      }
    },
    [],
  );

  const switchMode = useCallback(
    (newMode: "live" | "deepfake") => {
      setMode(newMode);
      sendCommand("switch_mode", { mode: newMode });
    },
    [sendCommand],
  );

  const toggleGradcam = useCallback(
    (enabled: boolean) => {
      setGradcamEnabled(enabled);
      sendCommand("toggle_gradcam", { enabled });
    },
    [sendCommand],
  );

  const addScoreEntry = useCallback((score: number) => {
    const now = new Date();
    const time = `${now.getMinutes()}:${now.getSeconds().toString().padStart(2, "0")}`;
    setScoreHistory((prev) => {
      const updated = [...prev, { time, score }];
      return updated.slice(-MAX_HISTORY_POINTS);
    });
  }, []);

  return {
    mode,
    gradcamEnabled,
    scoreHistory,
    switchMode,
    toggleGradcam,
  };
}
