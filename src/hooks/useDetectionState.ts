import { useState, useCallback, useRef } from "react";

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
  addScoreEntry: (score: number) => void;
}

const MAX_HISTORY_POINTS = 120; // 60 seconds at ~2 points/sec
const WS_URL = "ws://localhost:9734";

/**
 * State management hook for detection settings and score history.
 * Sends control commands to the Python backend via a lazily-opened
 * WebSocket (avoids opening a duplicate persistent connection).
 */
export function useDetectionState(): DetectionState {
  const [mode, setMode] = useState<"live" | "deepfake">("live");
  const [gradcamEnabled, setGradcamEnabled] = useState(false);
  const [scoreHistory, setScoreHistory] = useState<ScoreHistoryEntry[]>([]);
  const commandWsRef = useRef<WebSocket | null>(null);

  const getCommandWs = useCallback((): WebSocket | null => {
    if (
      commandWsRef.current &&
      commandWsRef.current.readyState === WebSocket.OPEN
    ) {
      return commandWsRef.current;
    }
    try {
      const ws = new WebSocket(WS_URL);
      commandWsRef.current = ws;
      ws.onclose = () => {
        commandWsRef.current = null;
      };
      return ws;
    } catch {
      return null;
    }
  }, []);

  const sendCommand = useCallback(
    (command: string, params: Record<string, unknown> = {}) => {
      const ws = getCommandWs();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command, ...params }));
      }
    },
    [getCommandWs],
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
    addScoreEntry,
  };
}
