import dataclasses


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclasses.dataclass(frozen=True)
class AngleController:
    target: float
    gamma: float = 1.0
    min_ratio: float = 0.5
    max_ratio: float = 2.0
    initial_step: float = 0.05
    min_step: float = 0.004
    max_step: float = 0.25

    def next_step(self, step: float, angle: float) -> float:
        ratio = (self.target / max(angle, 1e-12)) ** self.gamma
        ratio = clamp(ratio, self.min_ratio, self.max_ratio)
        return clamp(step * ratio, self.min_step, self.max_step)
