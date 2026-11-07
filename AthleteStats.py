from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


ACTIVITY_WINDOW_SECONDS = 3.0
EXPLOSIVE_MIN_SPEED = 0.05
EXPLOSIVE_MULTIPLIER = 1.5

TAKEDOWN_CONFIRM_SECONDS = 3.0
MIN_TRACKED_MOVEMENT = 0.02
STATE_CONFIRM_SECONDS = 0.2


@dataclass
class AthleteStats:
    """Statistics accumulated for one athlete."""

    athlete_id: int

    speed_samples: List[float] = field(default_factory=list)
    window_size: int = 30

    standuptime: float = 0.0
    partertime: float = 0.0
    unknown_posture_time: float = 0.0

    explosive_moments: List[Dict[str, float]] = field(default_factory=list)

    _activity_samples: List[Tuple[float, float]] = field(
        default_factory=list
    )

    _last_timestamp: Optional[float] = None
    _last_posture: Optional[str] = None
    _confirmed_posture: Optional[str] = None
    _pending_posture: Optional[str] = None
    _pending_since: Optional[float] = None

    _speed_sum: float = 0.0
    _speed_count: int = 0

    _in_explosive_moment: bool = False

    # Takedown tracking
    _takedown_attempt_active: bool = False
    _takedown_attempt_start: Optional[float] = None

    takedown_count: int = 0

    # Optional event history
    takedown_events: List[Dict[str, float]] = field(
        default_factory=list
    )

    def add_observation(
        self,
        timestamp: float,
        speed: float,
        posture: str
    ) -> Tuple[float, bool, bool]:

        """
        Process one observation.

        Returns:
            activity_average,
            is_explosive,
            takedown_confirmed
        """

        # ----------------------------------
        # 1. Calculate posture duration
        # ----------------------------------

        if self._last_timestamp is not None:
            if timestamp > self._last_timestamp:

                elapsed = timestamp - self._last_timestamp

                # A posture that has not survived the confirmation window is
                # deliberately charged to the preceding confirmed posture.
                if self._confirmed_posture == "standing":
                    self.standuptime += elapsed

                elif self._confirmed_posture == "ground":
                    self.partertime += elapsed

                else:
                    self.unknown_posture_time += elapsed

        # ----------------------------------
        # 2. Track speed
        # ----------------------------------

        baseline_average = (
            self._speed_sum / self._speed_count
            if self._speed_count
            else 0.0
        )

        self._speed_sum += speed
        self._speed_count += 1

        self.speed_samples.append(speed)

        if len(self.speed_samples) > self.window_size:
            self.speed_samples.pop(0)

        self._activity_samples.append((timestamp, speed))

        cutoff = timestamp - ACTIVITY_WINDOW_SECONDS

        self._activity_samples = [
            sample
            for sample in self._activity_samples
            if sample[0] >= cutoff
        ]

        activity_average = (
            sum(sample[1] for sample in self._activity_samples)
            / len(self._activity_samples)
            if self._activity_samples
            else 0.0
        )

        threshold = max(
            EXPLOSIVE_MIN_SPEED,
            baseline_average * EXPLOSIVE_MULTIPLIER
        )

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

        # ----------------------------------
        # 3. Takedown detection
        # ----------------------------------

        takedown_confirmed = False

        previous_posture = self._confirmed_posture

        if posture == self._confirmed_posture:
            self._pending_posture = None
            self._pending_since = None
        elif self._pending_posture is None:
            self._pending_posture = posture
            self._pending_since = timestamp
        elif self._pending_posture == posture:
            if timestamp - self._pending_since >= STATE_CONFIRM_SECONDS:
                self._confirmed_posture = posture
                self._pending_posture = None
                self._pending_since = None
        else:
            self._pending_posture = posture
            self._pending_since = timestamp

        confirmed_posture = self._confirmed_posture

        # Start attempt only on standing -> ground
        if (
            previous_posture == "standing"
            and confirmed_posture == "ground"
            and not self._takedown_attempt_active
        ):

            self._takedown_attempt_active = True
            self._takedown_attempt_start = timestamp

        # Check active attempt
        if self._takedown_attempt_active:

            # Athlete returned to standing before confirmation
            if confirmed_posture == "standing":

                self._takedown_attempt_active = False
                self._takedown_attempt_start = None

            elif confirmed_posture == "ground":

                elapsed_ground_time = (
                    timestamp - self._takedown_attempt_start
                )

                # Confirm after 3 seconds in ground
                if elapsed_ground_time >= TAKEDOWN_CONFIRM_SECONDS:

                    self.takedown_count += 1

                    self.takedown_events.append({
                        "timestamp": timestamp,
                        "duration": elapsed_ground_time,
                    })

                    takedown_confirmed = True

                    # Prevent duplicate counting
                    self._takedown_attempt_active = False
                    self._takedown_attempt_start = None

        # ----------------------------------
        # 4. Save current observation
        # ----------------------------------

        self._last_timestamp = timestamp
        # The very first observed posture is immediately established as the
        # baseline; confirmation is only needed for a transition.
        if self._confirmed_posture is None:
            self._confirmed_posture = posture
            self._pending_posture = None
            self._pending_since = None
        self._last_posture = self._confirmed_posture

        return (
            activity_average,
            is_explosive,
            takedown_confirmed
        )

    def get_confirmed_posture(self) -> str:
        """Return the stable posture used for timing and event detection."""
        return self._confirmed_posture or "unknown"

    def get_average_speed(self) -> float:

        return (
            self._speed_sum / self._speed_count
            if self._speed_count
            else 0.0
        )

    def to_dict(self) -> Dict[str, object]:

        classified_time = self.standuptime + self.partertime

        return {
            "person_id": self.athlete_id,

            "average_speed": self.get_average_speed(),

            "speed_unit": "normalized_frame_units_per_second",

            "standup_time_seconds": self.standuptime,
            "parter_time_seconds": self.partertime,
            "unknown_posture_time_seconds": self.unknown_posture_time,

            "standup_share": (
                self.standuptime / classified_time
                if classified_time
                else 0.0
            ),

            "parter_share": (
                self.partertime / classified_time
                if classified_time
                else 0.0
            ),

            "explosive_moment_count": len(self.explosive_moments),
            "explosive_moments": self.explosive_moments,

            "takedown_count": self.takedown_count,
            "takedown_events": self.takedown_events,
        }
