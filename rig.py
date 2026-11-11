import dataclasses
import pathlib
from typing import Any, Callable

import numpy as np
import torch

from _bootstrap import COMFY_ROOT, ROOT

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd

MODELS: pathlib.Path = COMFY_ROOT / "models"
DUMPS: pathlib.Path = ROOT / "dumps"
MEMORY_FRACTION: float = 0.93
NEGATIVE: str = "worst quality, low quality,"
PROMPTS: dict[str, str] = {
    "flat": (
        "1girl, flat color illustration, plain pastel background, "
        "simple shading"
    ),
    "face": "extreme close-up portrait of a girl's face, soft skin, "
    "detailed eyes",
    "fur": "dense forest foliage, fur texture, thousands of leaves",
    "grid": "checkerboard tiled floor, perspective grid, geometric "
    "architecture",
}

SamplerFunction = Callable[..., torch.Tensor]


@dataclasses.dataclass(frozen=True)
class RigConfig:
    unet: str = "anima_aestheticV11(1).safetensors"
    clip: str = "anima_baseV10_txt.safetensors"
    width: int = 576
    height: int = 768
    cfg: float = 5.0
    scheduler: str = "normal"


def limit_gpu_memory(fraction: float = MEMORY_FRACTION) -> None:
    torch.cuda.set_per_process_memory_fraction(fraction)


class Rig:
    def __init__(
        self,
        config: RigConfig = RigConfig(),
        prompts: dict[str, str] = PROMPTS,
    ) -> None:
        self.config: RigConfig = config
        self.patcher = comfy.sd.load_diffusion_model(
            str(MODELS / "diffusion_models" / config.unet),
            disable_dynamic=True,
        )
        clip = comfy.sd.load_clip(
            [str(MODELS / "clip" / config.clip)],
            clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
        )
        self.conditioning: dict[str, tuple[Any, Any]] = {
            name: (self.encode(clip, text), self.encode(clip, NEGATIVE))
            for name, text in prompts.items()
        }
        del clip
        comfy.model_management.soft_empty_cache()

    @staticmethod
    def encode(clip: Any, text: str) -> Any:
        return clip.encode_from_tokens_scheduled(clip.tokenize(text))

    def sigmas(self, steps: int, scheduler: str | None = None) -> torch.Tensor:
        sampling = self.patcher.get_model_object("model_sampling")
        return comfy.samplers.calculate_sigmas(
            sampling, scheduler or self.config.scheduler, steps
        )

    def empty_latent(self) -> torch.Tensor:
        latent = torch.zeros(
            [1, 4, self.config.height // 8, self.config.width // 8]
        )
        return comfy.sample.fix_empty_latent_channels(
            self.patcher, latent, downscale_ratio_spacial=8
        )

    def sample(
        self,
        prompt: str,
        seed: int,
        sigmas: torch.Tensor,
        sampler: SamplerFunction,
        cfg: float | None = None,
    ) -> torch.Tensor:
        latent = self.empty_latent()
        noise = comfy.sample.prepare_noise(latent, seed)
        positive, negative = self.conditioning[prompt]
        guider = comfy.samplers.CFGGuider(self.patcher)
        guider.set_conds(positive, negative)
        guider.set_cfg(self.config.cfg if cfg is None else cfg)
        result = guider.sample(
            noise,
            latent,
            comfy.samplers.KSAMPLER(sampler),
            sigmas,
            seed=seed,
            disable_pbar=True,
        )
        return result.detach().float().cpu()


def to_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()
