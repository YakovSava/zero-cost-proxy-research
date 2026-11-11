import dataclasses
import math

import torch


@dataclasses.dataclass(frozen=True)
class GaussianMixtureFlow:
    means: torch.Tensor
    scale: float
    weights: torch.Tensor

    @property
    def dimension(self) -> int:
        return self.means.shape[1]

    def variance(self, t: float) -> float:
        return (t * self.scale) ** 2 + (1.0 - t) ** 2

    def velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        variance: float = self.variance(t)
        offsets: torch.Tensor = x[:, None, :] - t * self.means[None]
        squared: torch.Tensor = offsets.pow(2).sum(dim=-1)
        logits: torch.Tensor = self.weights.log() - 0.5 * squared / variance
        posterior: torch.Tensor = torch.softmax(logits, dim=-1)
        gain: float = (t * self.scale**2 - (1.0 - t)) / variance
        per_mode: torch.Tensor = self.means[None] + gain * offsets
        return (posterior[..., None] * per_mode).sum(dim=1)

    def sample_noise(self, count: int, seed: int) -> torch.Tensor:
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        shape: tuple[int, int] = (count, self.dimension)
        return torch.randn(shape, generator=generator, dtype=torch.float64)


def random_mixture(
    dimension: int,
    modes: int,
    separation: float,
    scale: float,
    seed: int,
) -> GaussianMixtureFlow:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    directions: torch.Tensor = torch.randn(
        (modes, dimension), generator=generator, dtype=torch.float64
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    weights: torch.Tensor = torch.rand(
        modes, generator=generator, dtype=torch.float64
    ) + 0.5
    return GaussianMixtureFlow(
        means=separation * math.sqrt(dimension) / 4.0 * directions,
        scale=scale,
        weights=weights / weights.sum(),
    )


def isotropic_gaussian(dimension: int, scale: float) -> GaussianMixtureFlow:
    return GaussianMixtureFlow(
        means=torch.zeros((1, dimension), dtype=torch.float64),
        scale=scale,
        weights=torch.ones(1, dtype=torch.float64),
    )
