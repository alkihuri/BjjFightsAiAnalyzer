from dataclasses import dataclass, field
from typing import Dict, List, Tuple


ACTIVITY_WINDOW_SECONDS = 3.0
MIN_TRACKED_MOVEMENT = 0.65
EXPLOSIVE_MIN_SPEED = 0.03
EXPLOSIVE_MULTIPLIER = 1.5


@dataclass
class AthleteStats:
    """Per-athlete statistics accumulated only on analyzed video frames."""

    athlete_id: int
    speed_samples: List[float] = field(default_factory=list)
    window_size: int = 30
    standuptime: float = 0.0
    partertime: float = 0.0
    unknown_posture_time: float = 0.0
    explosive_moments: List[Dict[str, float]] = field(default_factory=list)
    _activity_samples: List[Tuple[float, float]] = field(default_factory=list)
    _last_timestamp: float | None = None
    _speed_sum: float = 0.0
    _speed_count: int = 0
    _in_explosive_moment: bool = False

    def add_observation(self, timestamp: float, speed: float, posture: str) -> Tuple[float, bool, bool]:
        """Store valid movement and return activity average, peak, and tracking flag."""
        if self._last_timestamp is not None and timestamp > self._last_timestamp:
            elapsed = timestamp - self._last_timestamp
            if posture == "standing":
                self.standuptime += elapsed
            elif posture == "ground":
                self.partertime += elapsed
            else:
                self.unknown_posture_time += elapsed
        self._last_timestamp = timestamp

        # Keep posture timing independent of speed, but do not let near-static
        # detections distort movement statistics or activity peaks.
        cutoff = timestamp - ACTIVITY_WINDOW_SECONDS
        self._activity_samples = [sample for sample in self._activity_samples if sample[0] >= cutoff]
        is_tracked_movement = speed >= MIN_TRACKED_MOVEMENT
        if not is_tracked_movement:
            activity_average = (
                sum(sample[1] for sample in self._activity_samples) / len(self._activity_samples)
                if self._activity_samples else 0.0
            )
            return activity_average, False, False

        baseline_average = self._speed_sum / self._speed_count if self._speed_count else 0.0
        self._speed_sum += speed
        self._speed_count += 1
        self.speed_samples.append(speed)
        if len(self.speed_samples) > self.window_size:
            self.speed_samples.pop(0)

        self._activity_samples.append((timestamp, speed))
        activity_average = sum(sample[1] for sample in self._activity_samples) / len(self._activity_samples)

        threshold = max(EXPLOSIVE_MIN_SPEED, baseline_average * EXPLOSIVE_MULTIPLIER)
        is_explosive = (
            self._speed_count >= 5
            and activity_average >= threshold
            and not self._in_explosive_moment
        )
        if is_explosive:
            self.explosive_moments.append({
                "timestamp": timestamp,
                "activity_average_speed": activity_average,
            })
            self._in_explosive_moment = True
        elif activity_average < threshold * 0.8:
            self._in_explosive_moment = False

        return activity_average, is_explosive, True

    def get_average_speed(self) -> float:
        return self._speed_sum / self._speed_count if self._speed_count else 0.0

    def to_dict(self) -> Dict[str, object]:
        classified_time = self.standuptime + self.partertime
        return {
            "person_id": self.athlete_id,
            "average_speed": self.get_average_speed(),
            "speed_unit": "normalized_frame_units_per_second",
            "standup_time_seconds": self.standuptime,
            "parter_time_seconds": self.partertime,
            "unknown_posture_time_seconds": self.unknown_posture_time,
            "standup_share": self.standuptime / classified_time if classified_time else 0.0,
            "parter_share": self.partertime / classified_time if classified_time else 0.0,
            "explosive_moment_count": len(self.explosive_moments),
            "explosive_moments": self.explosive_moments,
        }
