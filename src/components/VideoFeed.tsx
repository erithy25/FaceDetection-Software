import { useEffect, useRef } from "react";

interface VideoFeedProps {
  frame: string | null;
}

/**
 * Canvas-based video renderer. Receives base64-encoded JPEG frames
 * from the Python backend via WebSocket and renders them on an
 * HTML5 Canvas element using requestAnimationFrame for smooth display.
 */
export default function VideoFeed({ frame }: VideoFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!frame) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!imageRef.current) {
      imageRef.current = new Image();
    }

    const img = imageRef.current;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frame}`;
  }, [frame]);

  return (
    <div className="flex h-full w-full items-center justify-center bg-background">
      {frame ? (
        <canvas ref={canvasRef} className="video-canvas" />
      ) : (
        <div className="flex flex-col items-center gap-3 text-text-secondary">
          <svg
            className="h-16 w-16 opacity-30"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          <p className="text-sm">Waiting for video feed...</p>
          <p className="text-xs opacity-50">Start detection to begin capturing</p>
        </div>
      )}
    </div>
  );
}
