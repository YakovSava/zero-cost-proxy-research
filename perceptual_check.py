import functools
import json
import pathlib

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import profiles
import rig
import run_replica
import samplers
import units

import comfy.sd
import comfy.utils

FAMILIES: tuple[str, ...] = ("normal", "propagated1")
STEPS: tuple[int, ...] = (8, 12, 16)
VAE: str = "qwenImage_qwenImageVAE.safetensors"
OUTPUT: pathlib.Path = rig.ROOT / "results" / "perceptual.json"
IMAGES: pathlib.Path = rig.DUMPS / "images"
SHEETS: tuple[units.Unit, ...] = (
    units.Unit("face", 42),
    units.Unit("fur", 42),
)
TILE: int = 280
FONT_PATH: str = "C:/Windows/Fonts/arial.ttf"


def decode(vae: comfy.sd.VAE, latent: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        images = vae.decode_tiled(latent, tile_x=32, tile_y=32, overlap=8)
    return images.reshape(-1, *images.shape[-3:]).clamp(0.0, 1.0).float()


def lpips_input(images: torch.Tensor) -> torch.Tensor:
    return images.permute(0, 3, 1, 2) * 2.0 - 1.0


def to_picture(images: torch.Tensor) -> Image.Image:
    array = (images[0].numpy() * 255.0).round().astype(np.uint8)
    picture = Image.fromarray(array)
    height = round(picture.height * TILE / picture.width)
    return picture.resize((TILE, height), Image.LANCZOS)


def family_times(
    family: str,
    steps: int,
    own: profiles.Profile,
    others: list[profiles.Profile],
) -> np.ndarray:
    rate = run_replica.static_turning_density(others, 1.0)
    weight = run_replica.load_propagation(own.times) * rate
    return profiles.build_grid(own.times, weight, steps)


def caption(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str = "black",
) -> None:
    draw.text(position, text, fill=fill, font=font)


def load_picture(name: str) -> Image.Image:
    return to_picture(torch.from_numpy(np.load(IMAGES / f"{name}.npy")))


def sheet(unit: units.Unit, scores: dict[str, float]) -> None:
    font = ImageFont.truetype(FONT_PATH, 14)
    reference = load_picture(f"{unit.name}_reference")
    width = 110 + TILE * (len(STEPS) + 1)
    canvas = Image.new("RGB", (width, 24 + reference.height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for column, steps in enumerate(STEPS):
        caption(draw, (120 + column * TILE, 4), f"{steps} steps", font)
    caption(draw, (120 + len(STEPS) * TILE, 4), "reference", font)
    for row, family in enumerate(FAMILIES):
        top = 24 + row * reference.height
        caption(draw, (6, top + reference.height // 2), family, font)
        for column, steps in enumerate(STEPS):
            picture = load_picture(f"{unit.name}_{family}_{steps}")
            score = scores[f"{unit.name}|{family}|{steps}"]
            inner = ImageDraw.Draw(picture)
            inner.rectangle(
                (0, picture.height - 22, 110, picture.height), fill="black"
            )
            caption(
                inner,
                (4, picture.height - 20),
                f"LPIPS {score:.3f}",
                font,
                "white",
            )
            canvas.paste(picture, (110 + column * TILE, top))
        canvas.paste(reference, (110 + len(STEPS) * TILE, top))
    canvas.save(rig.ROOT / "figures" / f"grid_{unit.name}.jpg", quality=88)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    everything = units.all_units(list(rig.PROMPTS))
    loaded = {unit.name: profiles.load_profile(unit) for unit in everything}
    rig.limit_gpu_memory()
    harness = rig.Rig()
    vae = comfy.sd.VAE(
        sd=comfy.utils.load_torch_file(str(rig.MODELS / "vae" / VAE))
    )
    network = lpips.LPIPS(net="alex", verbose=False)
    sampler = functools.partial(samplers.sample_fixed, order=1, disable=True)
    scores: dict[str, float] = (
        json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    )
    for unit in everything:
        own = loaded[unit.name]
        others = [v for k, v in loaded.items() if k != unit.name]
        reference_latent = torch.from_numpy(own.final)
        reference = decode(vae, reference_latent)
        np.save(IMAGES / f"{unit.name}_reference.npy", reference.numpy())
        for family in FAMILIES:
            for steps in STEPS:
                key = f"{unit.name}|{family}|{steps}"
                if key in scores:
                    continue
                if family == "normal":
                    sigmas = harness.sigmas(steps)
                else:
                    times = family_times(family, steps, own, others)
                    sigmas = torch.tensor(
                        units.sigmas_from_times(times), dtype=torch.float32
                    )
                latent = harness.sample(
                    unit.prompt, unit.seed, sigmas, sampler
                )
                images = decode(vae, latent)
                np.save(
                    IMAGES / f"{unit.name}_{family}_{steps}.npy",
                    images.numpy(),
                )
                with torch.no_grad():
                    scores[key] = float(
                        network(lpips_input(images), lpips_input(reference))
                    )
        OUTPUT.write_text(json.dumps(scores))
        print(unit.name, flush=True)
    for unit in SHEETS:
        sheet(unit, scores)


if __name__ == "__main__":
    main()
