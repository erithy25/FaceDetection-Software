"""
SilentWitness — Main entry point for the Python ML backend.

Runs a WebSocket server on ws://localhost:9734 that:
1. Captures video frames from webcam or file input
2. Runs the face detection + classification pipeline
3. Streams results (frames, scores, features) to the Tauri frontend

Can be run standalone for development:
    python main.py                        # webcam mode
    python main.py --source video.mp4     # file mode
    python main.py --deepfake demo.mp4    # set deepfake file for demo switch
    python main.py --no-webcam            # start without opening webcam
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import base64
import time
from typing import Optional

import cv2
import numpy as np

# Ensure imports work whether run from python/ dir or project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets
from websockets.server import WebSocketServerProtocol

from capture import VideoCapture
from detector import FaceDetector
from classifier import DeepfakeClassifier
from scorer import ScoreFusion
from gradcam import GradCAM

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

    def __init__(
        self,
        source: Optional[str] = None,
        deepfake_file: Optional[str] = None,
        no_webcam: bool = False,
    ) -> None:
        self.capture = VideoCapture()
        self.detector = FaceDetector()
        self.classifier = DeepfakeClassifier()
        self.scorer = ScoreFusion()
        self.gradcam = GradCAM()

        self.running = False
        self.mode: str = "live"
        self.gradcam_enabled = False
        self.frame_count = 0

        self._initial_source = source
        self._no_webcam = no_webcam
        self._gradcam_loaded = False

        if deepfake_file:
            self.capture.set_deepfake_file(deepfake_file)

    async def start(self) -> None:
        """Initialize all pipeline components."""
        self.classifier.load_models()

        # Try loading Grad-CAM model (PyTorch version)
        gradcam_model = os.path.join(
            os.path.dirname(__file__), "models", "efficientnet_b0_deepfake.pt"
        )
        if os.path.exists(gradcam_model):
            self._gradcam_loaded = self.gradcam.load_model(gradcam_model)

        # Open video source
        if self._initial_source:
            opened = self.capture.open(self._initial_source)
        elif not self._no_webcam:
            opened = self.capture.open()
        else:
            opened = True  # No capture needed initially
            logger.info("Started without webcam (--no-webcam)")

        if not opened and not self._no_webcam:
            logger.warning(
                "Failed to open video source. The pipeline will wait for "
                "a source to become available."
            )

        self.running = True
        self.frame_count = 0
        self.scorer.reset()
        self.classifier.reset()
        logger.info("Detection pipeline started")

    async def stop(self) -> None:
        """Release all pipeline resources."""
        self.running = False
        self.capture.release()
        self.gradcam.cleanup()
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
        try:
            face_result = self.detector.detect(frame)
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            face_result = None

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

        # Frame-level classification (skip every other frame for performance)
        skip = self.frame_count % 2 != 0
        if skip and self.classifier.last_embedding is not None:
            frame_score = self.classifier.last_frame_score
            embedding = self.classifier.last_embedding
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
                "texture": round(frame_score, 4),
                "temporal": round(temporal_score, 4),
            },
        }

        # Optional Grad-CAM heatmap
        if self.gradcam_enabled and not skip and self._gradcam_loaded:
            heatmap = self.gradcam.generate(face_crop)
            if heatmap is not None:
                _, heatmap_png = cv2.imencode(".png", heatmap)
                result["heatmap"] = base64.b64encode(
                    heatmap_png.tobytes()
                ).decode("ascii")

        return result

    def handle_command(self, command: str, params: dict) -> dict:
        """Handle a command from the frontend. Returns a response dict."""
        if command == "switch_mode":
            new_mode = params.get("mode", "live")
            self.mode = new_mode
            if new_mode == "deepfake":
                self.capture.switch_to_file()
            else:
                self.capture.switch_to_webcam()
            # Reset scorer on mode switch for clean detection timing
            self.scorer.reset()
            self.classifier.reset()
            logger.info(f"Mode switched to: {new_mode}")
            return {"status": "ok", "mode": new_mode}

        elif command == "toggle_gradcam":
            self.gradcam_enabled = params.get("enabled", False)
            logger.info(
                f"Grad-CAM: {'enabled' if self.gradcam_enabled else 'disabled'}"
            )
            return {"status": "ok", "gradcam": self.gradcam_enabled}

        elif command == "set_deepfake_file":
            path = params.get("path", "")
            if path and os.path.exists(path):
                self.capture.set_deepfake_file(path)
                logger.info(f"Deepfake file set: {path}")
                return {"status": "ok", "path": path}
            return {"status": "error", "message": f"File not found: {path}"}

        elif command == "ping":
            return {"status": "pong"}

        else:
            logger.warning(f"Unknown command: {command}")
            return {"status": "error", "message": f"Unknown command: {command}"}


async def handler(
    websocket: WebSocketServerProtocol, pipeline: DetectionPipeline
) -> None:
    """Handle a single WebSocket connection from the Tauri frontend."""
    logger.info(f"Client connected: {websocket.remote_address}")

    try:
        await pipeline.start()

        async def receive_commands() -> None:
            """Listen for commands from the frontend."""
            try:
                async for message in websocket:
                    try:
                        msg = json.loads(message)
                        command = msg.pop("command", "")
                        if command:
                            response = pipeline.handle_command(command, msg)
                            await websocket.send(
                                json.dumps({"type": "response", **response})
                            )
                    except json.JSONDecodeError:
                        logger.warning("Received malformed command")
            except websockets.exceptions.ConnectionClosed:
                pass

        async def send_frames() -> None:
            """Continuously process and send frames."""
            while pipeline.running:
                try:
                    result = pipeline.process_frame()
                    if result is not None:
                        await websocket.send(json.dumps(result))
                    else:
                        # No frame available, wait a bit
                        await asyncio.sleep(0.033)  # ~30 FPS rate
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    await asyncio.sleep(0.1)
                # Yield control to allow command processing
                await asyncio.sleep(0.001)

        # Run both tasks concurrently
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(receive_commands()),
                asyncio.create_task(send_frames()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Handler error: {e}")
    finally:
        await pipeline.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SilentWitness — AI-Powered Deepfake Detection Backend"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: webcam index (int) or file path (default: webcam 0)",
    )
    parser.add_argument(
        "--deepfake",
        type=str,
        default=None,
        help="Path to deepfake video file for demo mode switching",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=WS_HOST,
        help=f"WebSocket host (default: {WS_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=WS_PORT,
        help=f"WebSocket port (default: {WS_PORT})",
    )
    parser.add_argument(
        "--no-webcam",
        action="store_true",
        help="Start without opening webcam (wait for frontend command)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main() -> None:
    """Start the WebSocket server."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse source (might be an int for webcam index)
    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass  # Keep as string (file path)

    pipeline = DetectionPipeline(
        source=source,
        deepfake_file=args.deepfake,
        no_webcam=args.no_webcam,
    )

    logger.info(
        f"Starting SilentWitness backend on ws://{args.host}:{args.port}"
    )

    stop_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    async with websockets.serve(
        lambda ws: handler(ws, pipeline),
        args.host,
        args.port,
        ping_interval=20,
        ping_timeout=60,
    ):
        logger.info("Backend ready, waiting for connections...")
        await stop_event.wait()

    logger.info("Backend shutting down")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
