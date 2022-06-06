import os
import pathlib

import numpy as np
import skimage
from PIL import Image
from skimage import exposure
from skimage import io
from skimage.filters import gaussian
from skimage.io._plugins.pil_plugin import pil_to_ndarray

from colors import print_blue


class Noise:
    GAUSSIAN = "gaussian"
    LOCALVAR = "localvar"
    POISSON = "poisson"
    SALT = "salt"
    PEPPER = "pepper"
    SnP = "s&p"
    SPECKLE = "speckle"


SIGMA_A = 1
SIGMA_B = 0.5
EXPOSURE = 0.0005  # 0.0005 seems to work well for the city images
NOISE = Noise.SPECKLE
NOISE_STRENGTH = 0.5
BLEND = 1  # lower values seem to work well for city images


def lerp(v0: float, v1: float, t: float) -> float:
    return (1 - t) * v0 + t * v1


def inv_lerp(a: float, b: float, v: float) -> float:
    return (v - a) / (b - a)


def do_gauss(im, sigma: float):
    if sigma <= 0:
        return im
    return gaussian(im, sigma=sigma, mode="reflect",
                    preserve_range=True, multichannel=True, truncate=4.0)


def load_and_apply(path: str, noise=NOISE, noise_strength=NOISE_STRENGTH,
                   clip_limit=EXPOSURE, blend=BLEND, sigma_a: float = SIGMA_A, sigma_b: float = SIGMA_B,
                   suffix='fx'):
    original = io.imread(str(path)) / 255.0
    processed = do_gauss(original, sigma_a)
    if noise is not None and noise_strength > 0:
        with_noise = skimage.util.random_noise(processed, mode=noise, clip=True)
        if noise_strength < 1:
            processed = np.ubyte(noise_strength * with_noise * 255 + (1 - noise_strength) * processed * 255)
        else:
            processed = with_noise
    if clip_limit > 0:
        processed = exposure.equalize_adapthist(processed, clip_limit=clip_limit)
    processed = do_gauss(processed, sigma_b)
    if BLEND < 1:
        processed = np.ubyte(blend * processed * 255 + (1 - blend) * original * 255)
    io.imsave(f"{path.split('.')[0]}_{suffix}.png", processed)


def apply_to_pil(pil_image: Image, output_path, save_name: str,
                 noise=NOISE, noise_strength=NOISE_STRENGTH,
                 clip_limit=EXPOSURE,
                 blend=BLEND, sigma_a: float = SIGMA_A, sigma_b: float = SIGMA_B):
    original = pil_to_ndarray(pil_image) / 255.0
    processed = do_gauss(original, sigma_a)
    if noise is not None and noise_strength > 0:
        with_noise = skimage.util.random_noise(processed, mode=noise, clip=True)
        if noise_strength < 1:
            processed = np.ubyte(noise_strength * with_noise * 255 + (1 - noise_strength) * processed * 255)
        else:
            processed = with_noise
    if clip_limit > 0:
        processed = exposure.equalize_adapthist(processed, clip_limit=clip_limit)
    processed = do_gauss(processed, sigma_b)
    if blend < 1:
        processed = np.ubyte(blend * processed * 255 + (1 - blend) * original * 255)
    fx_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    print_blue(f'saving {fx_path}')
    io.imsave(str(fx_path), processed)
