import { useState } from "react";

/**
 * Before/after frame comparison slider. Allows comparing the current
 * frame with the most suspicious frame (lowest confidence score) from
 * the current session. Uses a draggable divider for visual comparison.
 */
export default function FrameSlider() {
  const [sliderPos, setSliderPos] = useState(50);

  return (
    <div className="rounded-card border border-border bg-background p-3">
      <div
        className="relative h-40 w-full overflow-hidden rounded-lg bg-border/30"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - rect.left) / rect.width) * 100;
          setSliderPos(Math.max(0, Math.min(100, x)));
        }}
      >
        {/* Placeholder — frames will be populated by the detection state */}
        <div className="flex h-full items-center justify-center text-xs text-text-secondary">
          Frame comparison available during detection
        </div>

        {/* Slider divider */}
        <div
          className="absolute top-0 bottom-0 w-px bg-text-primary/50"
          style={{ left: `${sliderPos}%` }}
        >
          <div className="absolute top-1/2 left-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border border-text-primary/50 bg-surface" />
        </div>

        {/* Labels */}
        <div className="absolute bottom-2 left-2 rounded bg-background/80 px-1.5 py-0.5 text-[10px] text-text-secondary">
          Current
        </div>
        <div className="absolute bottom-2 right-2 rounded bg-background/80 px-1.5 py-0.5 text-[10px] text-text-secondary">
          Most Suspicious
        </div>
      </div>
    </div>
  );
}
