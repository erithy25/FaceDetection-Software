interface HeatmapOverlayProps {
  heatmap: string | null;
}

/**
 * Grad-CAM heatmap visualization overlay. Displays a semi-transparent
 * color map showing which regions of the face the classifier considers
 * most suspicious (blue = low attention, red = high attention).
 */
export default function HeatmapOverlay({ heatmap }: HeatmapOverlayProps) {
  if (!heatmap) return null;

  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <img
        src={`data:image/png;base64,${heatmap}`}
        alt="Grad-CAM heatmap"
        className="heatmap-overlay max-h-full max-w-full"
      />
    </div>
  );
}
