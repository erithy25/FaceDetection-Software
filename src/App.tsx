import { useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import VideoFeed from "./components/VideoFeed";
import ConfidenceRing from "./components/ConfidenceRing";
import StatusBadge from "./components/StatusBadge";
import ScoreGraph from "./components/ScoreGraph";
import HeatmapOverlay from "./components/HeatmapOverlay";
import FeatureBreakdown from "./components/FeatureBreakdown";
import DemoControls from "./components/DemoControls";
import FrameSlider from "./components/FrameSlider";
import { useWebSocket } from "./hooks/useWebSocket";
import { useDetectionState } from "./hooks/useDetectionState";

function App() {
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const {
    mode,
    gradcamEnabled,
    scoreHistory,
    switchMode,
    toggleGradcam,
    addScoreEntry,
  } = useDetectionState();
  const { frame, score, status, temporalScore, frameScore, fps, features, heatmap, connected } =
    useWebSocket("ws://localhost:9734", addScoreEntry);

  const handleStartStop = useCallback(async () => {
    setStartError(null);
    try {
      if (detecting) {
        await invoke("stop_detection");
        setDetecting(false);
      } else {
        await invoke("start_detection");
        setDetecting(true);
      }
    } catch (err) {
      setStartError(String(err));
    }
  }, [detecting]);

  return (
    <div className="flex h-screen w-screen flex-col bg-background font-sans text-text-primary">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-6 py-3">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold tracking-tight">SilentWitness</h1>
          <span className="rounded-full bg-surface px-2.5 py-0.5 text-xs text-text-secondary">
            v1.0
          </span>
          {/* Connection indicator */}
          <span
            className={`h-2 w-2 rounded-full transition-colors duration-300 ${
              connected ? "bg-accent-green" : "bg-accent-red"
            }`}
            title={connected ? "Backend connected" : "Backend disconnected"}
          />
        </div>
        <div className="flex items-center gap-3">
          {/* Start/Stop Detection Button */}
          <button
            onClick={handleStartStop}
            className={`rounded-button px-4 py-1.5 text-sm font-medium transition-colors duration-300 ${
              detecting
                ? "bg-accent-red/20 text-accent-red hover:bg-accent-red/30"
                : "bg-accent-green/20 text-accent-green hover:bg-accent-green/30"
            }`}
          >
            {detecting ? "Stop Detection" : "Start Detection"}
          </button>

          <button
            onClick={() => setShowAnalysis(!showAnalysis)}
            className="rounded-button border border-border bg-surface px-3 py-1.5 text-sm
                       text-text-secondary transition-colors duration-300 hover:text-text-primary"
          >
            {showAnalysis ? "Dashboard" : "Analysis"}
          </button>
          <DemoControls mode={mode} onSwitchMode={switchMode} currentStatus={status} />
        </div>
      </header>

      {/* Error banner */}
      {startError && (
        <div className="border-b border-accent-red/30 bg-accent-red/10 px-6 py-2">
          <span className="text-xs text-accent-red">{startError}</span>
          <button
            onClick={() => setStartError(null)}
            className="ml-3 text-xs text-text-secondary hover:text-text-primary"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Content */}
      <main className="flex flex-1 overflow-hidden">
        {/* Video Feed Section */}
        <div className="relative flex-1">
          <VideoFeed frame={frame} />

          {/* Score Overlay (top right) */}
          <div className="absolute right-4 top-4 flex flex-col items-center gap-2">
            <ConfidenceRing score={score} />
            <StatusBadge status={status} />
          </div>

          {/* FPS Counter (bottom left) */}
          <div className="absolute bottom-4 left-4">
            <span className="font-mono text-xs text-text-secondary">
              {fps.toFixed(1)} FPS
            </span>
          </div>

          {/* Heatmap Overlay */}
          {gradcamEnabled && heatmap && (
            <HeatmapOverlay heatmap={heatmap} />
          )}
        </div>

        {/* Analysis Panel (collapsible) */}
        {showAnalysis && (
          <aside className="w-96 border-l border-border bg-surface p-4 overflow-y-auto">
            <h2 className="mb-4 text-sm font-medium text-text-secondary uppercase tracking-wider">
              Analysis
            </h2>

            <div className="space-y-6">
              <FeatureBreakdown features={features} />

              <div>
                <h3 className="mb-2 text-xs font-medium text-text-secondary uppercase tracking-wider">
                  Score History (60s)
                </h3>
                <ScoreGraph scoreHistory={scoreHistory} />
              </div>

              <div>
                <h3 className="mb-2 text-xs font-medium text-text-secondary uppercase tracking-wider">
                  Frame Comparison
                </h3>
                <FrameSlider />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">Grad-CAM Overlay</span>
                <button
                  onClick={() => toggleGradcam(!gradcamEnabled)}
                  className={`relative h-6 w-11 rounded-full transition-colors duration-300 ${
                    gradcamEnabled ? "bg-accent-green" : "bg-border"
                  }`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform duration-300 ${
                      gradcamEnabled ? "translate-x-5" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>

              <div className="rounded-card border border-border bg-background p-3">
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-text-secondary">Frame Score</span>
                    <p className="font-mono text-sm font-semibold">{frameScore.toFixed(3)}</p>
                  </div>
                  <div>
                    <span className="text-text-secondary">Temporal Score</span>
                    <p className="font-mono text-sm font-semibold">{temporalScore.toFixed(3)}</p>
                  </div>
                </div>
              </div>
            </div>
          </aside>
        )}
      </main>
    </div>
  );
}

export default App;
