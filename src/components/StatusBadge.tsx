interface StatusBadgeProps {
  status: "authentic" | "suspicious" | "deepfake" | "standby" | "warming_up";
}

const STATUS_CONFIG = {
  authentic: {
    label: "AUTHENTIC",
    bg: "bg-accent-green/15",
    text: "text-accent-green",
    dot: "bg-accent-green",
  },
  suspicious: {
    label: "SUSPICIOUS",
    bg: "bg-accent-yellow/15",
    text: "text-accent-yellow",
    dot: "bg-accent-yellow",
  },
  deepfake: {
    label: "DEEPFAKE DETECTED",
    bg: "bg-accent-red/15",
    text: "text-accent-red",
    dot: "bg-accent-red",
  },
  standby: {
    label: "STANDBY",
    bg: "bg-border/50",
    text: "text-text-secondary",
    dot: "bg-text-secondary",
  },
  warming_up: {
    label: "WARMING UP",
    bg: "bg-border/50",
    text: "text-text-secondary",
    dot: "bg-text-secondary",
  },
} as const;

/**
 * Text status display showing the current detection result.
 * Uses color-coded backgrounds and an animated dot indicator.
 */
export default function StatusBadge({ status }: StatusBadgeProps) {
  const config = STATUS_CONFIG[status];

  return (
    <div
      className={`status-badge flex items-center gap-2 rounded-full px-3 py-1 ${config.bg}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${config.dot}`} />
      <span className={`text-xs font-semibold tracking-wide ${config.text}`}>
        {config.label}
      </span>
    </div>
  );
}
