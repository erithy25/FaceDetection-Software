"""
Score Fusion Module — Combines frame and temporal scores.

Implements weighted score fusion, exponential moving average (EMA)
smoothing, and hysteresis filtering to produce a stable, reliable
final authenticity score.
"""

from typing import Optional


class ScoreFusion:
    """Weighted score fusion with EMA smoothing and hysteresis."""

    # Fusion weights
    FRAME_WEIGHT = 0.6
    TEMPORAL_WEIGHT = 0.4

    # EMA smoothing factor (higher = more responsive, lower = smoother)
    EMA_ALPHA = 0.3

    # Status thresholds
    GREEN_THRESHOLD = 0.85
    RED_THRESHOLD = 0.60

    # Hysteresis: consecutive readings required to change status
    DOWNGRADE_COUNT = 5  # Slow to degrade (prevent false alarms)
    UPGRADE_COUNT = 2    # Fast to recover (quick alert clearance)

    def __init__(self) -> None:
        self._ema_score: Optional[float] = None
        self._raw_scores: list[float] = []

        # Hysteresis state
        self._current_status: str = "warming_up"
        self._pending_status: Optional[str] = None
        self._pending_count: int = 0

        # Cold start tracking
        self._total_frames: int = 0
        self._warming_up: bool = True

    def update(self, frame_score: float, temporal_score: float) -> float:
        """Update the fused score with new frame and temporal scores.

        Args:
            frame_score: Per-frame classifier output [0.0–1.0].
            temporal_score: LSTM temporal analysis output [0.0–1.0].

        Returns:
            Smoothed final score [0.0–1.0].
        """
        self._total_frames += 1

        # During cold start (first 16 frames), only use frame score
        if self._total_frames < 16:
            self._warming_up = True
            fused = frame_score
        else:
            self._warming_up = False
            fused = (
                self.FRAME_WEIGHT * frame_score
                + self.TEMPORAL_WEIGHT * temporal_score
            )

        # Apply EMA smoothing
        if self._ema_score is None:
            self._ema_score = fused
        else:
            self._ema_score = (
                self.EMA_ALPHA * fused + (1 - self.EMA_ALPHA) * self._ema_score
            )

        # Track raw scores for hysteresis
        self._raw_scores.append(self._ema_score)
        if len(self._raw_scores) > 10:
            self._raw_scores = self._raw_scores[-10:]

        # Update status with hysteresis
        self._update_status()

        return self._ema_score

    def get_status(self) -> str:
        """Get the current detection status string.

        Returns one of: "authentic", "suspicious", "deepfake",
        "standby", "warming_up".
        """
        if self._warming_up:
            return "warming_up"
        return self._current_status

    def get_score(self) -> float:
        """Get the current smoothed score."""
        return self._ema_score if self._ema_score is not None else 0.0

    def reset(self) -> None:
        """Reset all scoring state."""
        self._ema_score = None
        self._raw_scores = []
        self._current_status = "warming_up"
        self._pending_status = None
        self._pending_count = 0
        self._total_frames = 0
        self._warming_up = True

    def _update_status(self) -> None:
        """Apply hysteresis logic to determine status transitions."""
        if self._ema_score is None:
            return

        # Determine target status from current score
        if self._ema_score > self.GREEN_THRESHOLD:
            target = "authentic"
        elif self._ema_score >= self.RED_THRESHOLD:
            target = "suspicious"
        else:
            target = "deepfake"

        if target == self._current_status:
            # Score is in the same zone — reset pending transition
            self._pending_status = None
            self._pending_count = 0
            return

        # Determine if this is a downgrade or upgrade
        status_order = {"authentic": 2, "suspicious": 1, "deepfake": 0}
        is_downgrade = status_order.get(target, 0) < status_order.get(
            self._current_status, 0
        )
        required_count = self.DOWNGRADE_COUNT if is_downgrade else self.UPGRADE_COUNT

        if target == self._pending_status:
            self._pending_count += 1
        else:
            self._pending_status = target
            self._pending_count = 1

        if self._pending_count >= required_count:
            self._current_status = target
            self._pending_status = None
            self._pending_count = 0
