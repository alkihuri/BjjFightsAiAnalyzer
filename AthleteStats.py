from dataclasses import dataclass, field
from typing import List

@dataclass
class AthleteStats:
    athlete_id: int
    speed_samples: List[float] = field(default_factory=list)
    window_size: int = 30

    def add_speed(self, speed: float) -> None:
        self.speed_samples.append(speed)
        if len(self.speed_samples) > self.window_size:
            self.speed_samples.pop(0)

    def get_average_speed(self) -> float:
        if not self.speed_samples:
            return 0.0
        return sum(self.speed_samples) / len(self.speed_samples)

    def reset(self) -> None:
        self.speed_samples = []
