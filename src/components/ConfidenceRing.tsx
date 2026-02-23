interface ConfidenceRingProps {
  score: number;
}

/**
 * Animated circular confidence indicator. Displays the current
 * authenticity score as a colored ring that transitions smoothly
 * between green (authentic), yellow (suspicious), and red (deepfake).
 */
export default function ConfidenceRing({ score }: ConfidenceRingProps) {
  const radius = 40;
  const strokeWidth = 4;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - score);

  const getColor = (s: number): string => {
    if (s > 0.85) return "#34D399"; // accent-green
    if (s >= 0.60) return "#FBBF24"; // accent-yellow
    return "#F87171"; // accent-red
  };

  const color = getColor(score);

  return (
    <div className="confidence-ring">
      <svg width={96} height={96} viewBox="0 0 96 96">
        {/* Background ring */}
        <circle
          cx={48}
          cy={48}
          r={radius}
          fill="none"
          stroke="#2A2A2E"
          strokeWidth={strokeWidth}
        />
        {/* Score ring */}
        <circle
          cx={48}
          cy={48}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          className="confidence-ring-circle"
          transform="rotate(-90 48 48)"
        />
      </svg>
      {/* Score text */}
      <div className="absolute flex flex-col items-center">
        <span
          className="score-value font-mono text-xl font-semibold"
          style={{ color }}
        >
          {Math.round(score * 100)}
        </span>
        <span className="text-[10px] text-text-secondary">%</span>
      </div>
    </div>
  );
}
