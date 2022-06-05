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


BLUR_SIGMA = [1, 0.5]  # [1, 0.5] seems to work well for city images
EXPOSURE = 0.01  # 0.0005 seems to work well for the city images

NOISE = Noise.GAUSSIAN
NOISE_STRENGTH = 0.75
BLEND = 1  # lower values seem to work well for city images


def lerp(v0: float, v1: float, t: float) -> float:
    return (1 - t) * v0 + t * v1


def inv_lerp(a: float, b: float, v: float) -> float:
    return (v - a) / (b - a)


def do_gauss(im, index: int):
    return gaussian(im, sigma=BLUR_SIGMA[index], mode="reflect",
                    preserve_range=True, multichannel=True, truncate=4.0)


def load_and_apply(path: str):
    original = io.imread(str(path)) / 255.0
    processed = do_gauss(original, 0)
    processed = skimage.util.random_noise(processed, mode=NOISE, clip=True)
    processed = exposure.equalize_adapthist(processed, clip_limit=EXPOSURE)
    processed = do_gauss(processed, 1)
    if BLEND < 1:
        processed = np.ubyte(BLEND * processed * 255 + (1 - BLEND) * original * 255)
    io.imsave(f"{path.split('.')[0]}_fx.png", processed)


def apply_to_pil(pil_image: Image, output_path, save_name: str,
                 noise=NOISE, noise_strength=NOISE_STRENGTH, blend=BLEND):
    original = pil_to_ndarray(pil_image) / 255.0
    processed = do_gauss(original, 0)
    with_noise = skimage.util.random_noise(processed, mode=noise, clip=True)
    if noise_strength < 1:
        processed = np.ubyte(noise_strength * with_noise * 255 + (1 - noise_strength) * processed * 255)
    else:
        processed = with_noise
    processed = exposure.equalize_adapthist(processed, clip_limit=EXPOSURE)
    processed = do_gauss(processed, 1)
    if blend < 1:
        processed = np.ubyte(blend * processed * 255 + (1 - blend) * original * 255)
    fx_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    print_blue(f'saving {fx_path}')
    io.imsave(str(fx_path), processed)
