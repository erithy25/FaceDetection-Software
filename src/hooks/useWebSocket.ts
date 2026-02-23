import { useEffect, useRef, useState, useCallback } from "react";

export type DetectionStatus =
  | "authentic"
  | "suspicious"
  | "deepfake"
  | "standby"
  | "warming_up";

export interface Features {
  blink: number;
  texture: number;
  temporal: number;
}

interface DetectionMessage {
  type: "frame";
  data: string;
  score: number;
  status: DetectionStatus;
  temporal_score: number;
  frame_score: number;
  fps: number;
  features: Features;
  heatmap?: string;
}

interface WebSocketState {
  frame: string | null;
  score: number;
  status: DetectionStatus;
  temporalScore: number;
  frameScore: number;
  fps: number;
  features: Features;
  heatmap: string | null;
  connected: boolean;
}

const DEFAULT_FEATURES: Features = { blink: 0, texture: 0, temporal: 0 };

/**
 * WebSocket hook for receiving detection results and video frames
 * from the Python backend. Handles reconnection with exponential backoff.
 */
export function useWebSocket(url: string): WebSocketState {
  const [state, setState] = useState<WebSocketState>({
    frame: null,
    score: 0,
    status: "standby",
    temporalScore: 0,
    frameScore: 0,
    fps: 0,
    features: DEFAULT_FEATURES,
    heatmap: null,
    connected: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>();
  const reconnectDelay = useRef(1000);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((prev) => ({ ...prev, connected: true }));
        reconnectDelay.current = 1000;
      };

      ws.onmessage = (event) => {
        try {
          const msg: DetectionMessage = JSON.parse(event.data);
          if (msg.type === "frame") {
            setState({
              frame: msg.data,
              score: msg.score,
              status: msg.status as DetectionStatus,
              temporalScore: msg.temporal_score,
              frameScore: msg.frame_score,
              fps: msg.fps,
              features: msg.features ?? DEFAULT_FEATURES,
              heatmap: msg.heatmap ?? null,
              connected: true,
            });
          }
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }));
        reconnectTimeout.current = setTimeout(() => {
          reconnectDelay.current = Math.min(reconnectDelay.current * 2, 10000);
          connect();
        }, reconnectDelay.current);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      reconnectTimeout.current = setTimeout(connect, reconnectDelay.current);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    };
  }, [connect]);

  return state;
}
