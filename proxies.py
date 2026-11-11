import dataclasses

import torch

NORM_FLOOR: float = 1e-12


def flatten(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


@dataclasses.dataclass(frozen=True)
class Increment:
    speed_new: torch.Tensor
    speed_old: torch.Tensor
    angle: torch.Tensor
    gap: torch.Tensor

    @property
    def cosine(self) -> torch.Tensor:
        return torch.cos(self.angle)

    @property
    def rotation(self) -> torch.Tensor:
        product: torch.Tensor = self.speed_new * self.speed_old
        return torch.sqrt(product) * 2.0 * torch.sin(0.5 * self.angle)

    @property
    def stretch(self) -> torch.Tensor:
        return (self.speed_new - self.speed_old).abs()

    @property
    def mean_speed(self) -> torch.Tensor:
        return 0.5 * (self.speed_new + self.speed_old)


def increment(v_new: torch.Tensor, v_old: torch.Tensor) -> Increment:
    new: torch.Tensor = flatten(v_new)
    old: torch.Tensor = flatten(v_old)
    speed_new: torch.Tensor = new.norm(dim=1)
    speed_old: torch.Tensor = old.norm(dim=1)
    unit_new: torch.Tensor = new / speed_new.clamp_min(NORM_FLOOR)[:, None]
    unit_old: torch.Tensor = old / speed_old.clamp_min(NORM_FLOOR)[:, None]
    chord: torch.Tensor = (unit_new - unit_old).norm(dim=1)
    angle: torch.Tensor = 2.0 * torch.asin((0.5 * chord).clamp(max=1.0))
    return Increment(
        speed_new=speed_new,
        speed_old=speed_old,
        angle=angle,
        gap=(new - old).norm(dim=1),
    )


def acceleration(value: Increment, step: float) -> torch.Tensor:
    return value.gap / step


def rotation_rate(value: Increment, step: float) -> torch.Tensor:
    return value.rotation / step


def stretch_rate(value: Increment, step: float) -> torch.Tensor:
    return value.stretch / step


def curvature(value: Increment, step: float) -> torch.Tensor:
    arc: torch.Tensor = value.mean_speed * step
    return value.angle / arc.clamp_min(NORM_FLOOR)


def turning_rate(value: Increment, step: float) -> torch.Tensor:
    return value.angle / step
