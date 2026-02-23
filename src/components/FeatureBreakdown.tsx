interface Features {
  blink: number;
  texture: number;
  temporal: number;
}

interface FeatureBreakdownProps {
  features: Features;
}

const FEATURE_LABELS: { key: keyof Features; label: string }[] = [
  { key: "blink", label: "Blink Pattern" },
  { key: "texture", label: "Texture Consistency" },
  { key: "temporal", label: "Temporal Coherence" },
];

function getBarColor(value: number): string {
  if (value > 0.85) return "bg-accent-green";
  if (value >= 0.60) return "bg-accent-yellow";
  return "bg-accent-red";
}

/**
 * Displays individual detection feature scores as horizontal bar indicators.
 * Shows blink pattern, texture consistency, and temporal coherence.
 */
export default function FeatureBreakdown({ features }: FeatureBreakdownProps) {
  return (
    <div>
      <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-text-secondary">
        Feature Breakdown
      </h3>
      <div className="space-y-3">
        {FEATURE_LABELS.map(({ key, label }) => {
          const value = features[key];
          return (
            <div key={key}>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-xs text-text-secondary">{label}</span>
                <span className="font-mono text-xs font-medium text-text-primary">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-border">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${getBarColor(value)}`}
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
