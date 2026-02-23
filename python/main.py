"""
SilentWitness — Main entry point for the Python ML backend.

Runs a WebSocket server on ws://localhost:9734 that:
1. Captures video frames from webcam or file input
2. Runs the face detection + classification pipeline
3. Streams results (frames, scores, features) to the Tauri frontend
"""

import asyncio
import json
import logging
import signal
import base64
from typing import Optional

import cv2
import websockets
from websockets.server import WebSocketServerProtocol

from capture import VideoCapture
from detector import FaceDetector
from classifier import DeepfakeClassifier
from scorer import ScoreFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("silentwitness")

# WebSocket server configuration
WS_HOST = "localhost"
WS_PORT = 9734
JPEG_QUALITY = 80


class DetectionPipeline:
    """Orchestrates the full detection pipeline from capture to scoring."""

    def __init__(self) -> None:
        self.capture = VideoCapture()
        self.detector = FaceDetector()
        self.classifier = DeepfakeClassifier()
        self.scorer = ScoreFusion()

        self.running = False
        self.mode: str = "live"  # "live" or "deepfake"
        self.gradcam_enabled = False
        self.frame_count = 0

    async def start(self) -> None:
        """Initialize all pipeline components."""
        self.capture.open()
        self.classifier.load_models()
        self.running = True
        self.frame_count = 0
        logger.info("Detection pipeline started")

    async def stop(self) -> None:
        """Release all pipeline resources."""
        self.running = False
        self.capture.release()
        logger.info("Detection pipeline stopped")

    def process_frame(self) -> Optional[dict]:
        """Process a single frame through the full pipeline.

        Returns a dict with frame data, scores, and features,
        or None if no frame is available.
        """
        frame = self.capture.read()
        if frame is None:
            return None

        self.frame_count += 1

        # Face detection and preprocessing
        face_result = self.detector.detect(frame)
        if face_result is None:
            # No face detected — encode frame and return standby status
            _, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            return {
                "type": "frame",
                "data": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                "score": 0.0,
                "status": "standby",
                "temporal_score": 0.0,
                "frame_score": 0.0,
                "fps": self.capture.fps,
                "features": {"blink": 0.0, "texture": 0.0, "temporal": 0.0},
            }

        face_crop, landmarks, features = face_result

        # Frame-level classification (every frame or every 2nd frame)
        skip = self.frame_count % 2 != 0
        if skip:
            frame_score, embedding = self.classifier.last_frame_score, self.classifier.last_embedding
        else:
            frame_score, embedding = self.classifier.classify_frame(face_crop)

        # Temporal analysis (every 8 frames when buffer is full)
        temporal_score = self.classifier.analyze_temporal(embedding)

        # Score fusion with EMA and hysteresis
        fused = self.scorer.update(frame_score, temporal_score)

        # Determine status from fused score
        status = self.scorer.get_status()

        # Encode frame for display
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        result = {
            "type": "frame",
            "data": base64.b64encode(jpeg.tobytes()).decode("ascii"),
            "score": round(fused, 4),
            "status": status,
            "temporal_score": round(temporal_score, 4),
            "frame_score": round(frame_score, 4),
            "fps": round(self.capture.fps, 1),
            "features": {
                "blink": round(features.get("blink", 0.0), 4),
                "texture": round(frame_score, 4),  # Texture score derived from frame classifier
                "temporal": round(temporal_score, 4),
            },
        }

        # Optional Grad-CAM heatmap
        if self.gradcam_enabled and not skip:
            heatmap = self.classifier.generate_gradcam(face_crop)
            if heatmap is not None:
                _, heatmap_png = cv2.imencode(".png", heatmap)
                result["heatmap"] = base64.b64encode(
                    heatmap_png.tobytes()
                ).decode("ascii")

        return result

    def handle_command(self, command: str, params: dict) -> None:
        """Handle a command from the frontend."""
        if command == "switch_mode":
            new_mode = params.get("mode", "live")
            self.mode = new_mode
            if new_mode == "deepfake":
                self.capture.switch_to_file()
            else:
                self.capture.switch_to_webcam()
            logger.info(f"Mode switched to: {new_mode}")

        elif command == "toggle_gradcam":
            self.gradcam_enabled = params.get("enabled", False)
            logger.info(f"Grad-CAM: {'enabled' if self.gradcam_enabled else 'disabled'}")


pipeline = DetectionPipeline()


async def handler(websocket: WebSocketServerProtocol) -> None:
    """Handle a single WebSocket connection from the Tauri frontend."""
    logger.info(f"Client connected: {websocket.remote_address}")

    try:
        await pipeline.start()

        async def receive_commands() -> None:
            """Listen for commands from the frontend."""
            async for message in websocket:
                try:
                    msg = json.loads(message)
                    command = msg.get("command", "")
                    pipeline.handle_command(command, msg)
                except json.JSONDecodeError:
                    logger.warning("Received malformed command")

        async def send_frames() -> None:
            """Continuously process and send frames."""
            while pipeline.running:
                result = pipeline.process_frame()
                if result is not None:
                    await websocket.send(json.dumps(result))
                # Yield control to allow command processing
                await asyncio.sleep(0.001)

        # Run both tasks concurrently
        await asyncio.gather(
            receive_commands(),
            send_frames(),
            return_exceptions=True,
        )
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        await pipeline.stop()


async def main() -> None:
    """Start the WebSocket server."""
    logger.info(f"Starting SilentWitness backend on ws://{WS_HOST}:{WS_PORT}")

    stop = asyncio.get_event_loop().create_future()

    def signal_handler() -> None:
        stop.set_result(None)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    async with websockets.serve(handler, WS_HOST, WS_PORT):
        logger.info("Backend ready, waiting for connections...")
        await stop

    logger.info("Backend shutting down")


if __name__ == "__main__":
    asyncio.run(main())
