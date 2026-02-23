"""
Video Capture Module — Handles webcam and file-based video input.

Provides threaded frame acquisition to prevent pipeline stalls,
automatic resolution detection and downscaling, and mode switching
between live webcam and pre-recorded deepfake files.
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np

# Processing resolution (downscaled from native for performance)
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480
DEFAULT_FPS = 30
FRAME_BUFFER_SIZE = 30  # 1 second at 30 FPS


class VideoCapture:
    """Thread-separated video capture with automatic downscaling."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self.fps: float = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()

        # Default input sources
        self._webcam_index = 0
        self._deepfake_file: Optional[str] = None
        self._use_file = False

    def open(self, source: Optional[int | str] = None) -> bool:
        """Open a video source (webcam index or file path).

        If no source is provided, defaults to webcam 0.
        """
        if source is None:
            source = self._webcam_index

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            return False

        # Start the capture thread
        self._running = True
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def release(self) -> None:
        """Stop capture and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[np.ndarray]:
        """Read the most recent frame (thread-safe)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def switch_to_webcam(self) -> None:
        """Switch input to live webcam."""
        self._use_file = False
        self.release()
        self.open(self._webcam_index)

    def switch_to_file(self, path: Optional[str] = None) -> None:
        """Switch input to a pre-recorded deepfake video file."""
        if path is not None:
            self._deepfake_file = path
        self._use_file = True
        self.release()
        if self._deepfake_file:
            self.open(self._deepfake_file)

    def set_deepfake_file(self, path: str) -> None:
        """Set the path to the deepfake video file for demo mode."""
        self._deepfake_file = path

    def _capture_loop(self) -> None:
        """Background thread: continuously captures frames from the source."""
        while self._running and self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()

            if not ret:
                # If reading from file, loop back to start
                if self._use_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Downscale to processing resolution
            h, w = frame.shape[:2]
            if w != PROCESS_WIDTH or h != PROCESS_HEIGHT:
                frame = cv2.resize(
                    frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA
                )

            with self._lock:
                self._frame = frame

            # Update FPS counter
            self._frame_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = time.time()
