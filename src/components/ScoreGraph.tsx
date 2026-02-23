import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ReferenceArea,
  ResponsiveContainer,
} from "recharts";

interface ScoreHistoryEntry {
  time: string;
  score: number;
}

interface ScoreGraphProps {
  scoreHistory: ScoreHistoryEntry[];
}

/**
 * Temporal line chart showing score history over the last 60 seconds.
 * Background zones colored green (>0.85), yellow (0.60-0.85), red (<0.60).
 */
export default function ScoreGraph({ scoreHistory }: ScoreGraphProps) {
  return (
    <div className="rounded-card border border-border bg-background p-3">
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={scoreHistory} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          {/* Zone backgrounds */}
          <ReferenceArea y1={0.85} y2={1.0} fill="#34D399" fillOpacity={0.08} />
          <ReferenceArea y1={0.60} y2={0.85} fill="#FBBF24" fillOpacity={0.08} />
          <ReferenceArea y1={0.0} y2={0.60} fill="#F87171" fillOpacity={0.08} />

          <XAxis
            dataKey="time"
            tick={{ fill: "#8E8E93", fontSize: 10 }}
            axisLine={{ stroke: "#2A2A2E" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "#8E8E93", fontSize: 10 }}
            axisLine={{ stroke: "#2A2A2E" }}
            tickLine={false}
            width={30}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#F5F5F7"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
