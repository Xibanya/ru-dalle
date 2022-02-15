from datetime import datetime
import glob
import os

from PIL import Image
import colors
from rudalle import get_realesrgan
from rudalle.pipelines import super_resolution

FOLDER = 'content/output/City'

ROWS = 4
COLS = 4
SIZE = 256


def make_grid(folder: str = FOLDER,
              rows: int = ROWS,
              cols: int = COLS,
              size: int = SIZE,
              show: bool = False
              ):

    paths = []
    types = ('*.png', '*.jpg', "*.jpeg", "*.bmp")
    for ext in types:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    colors.print_blue(f"{len(paths)} images found")

    new = Image.new("RGB", (size * cols, size * rows))

    for y in range(rows):
        for x in range(cols):
            index = x + y * cols
            if index > len(paths) - 1:
                index = index % len(paths)
            img = Image.open(paths[index])
            new.paste(img, (x * size, y * size))

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_path = folder + f'/grid_{rows}x{cols}_{now}.png'
    new.save(save_path)
    colors.print_cyan(f'saved to {save_path}')
    if show:
        new.show()


def supersize_all(folder: str = FOLDER, model: str = 'x4'):
    paths = []
    images = []
    types = ('*.png', '*.jpg', "*.jpeg", "*.bmp")
    for ext in types:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    colors.print_blue(f"{len(paths)} images found")
    for path in paths:
        images.append(Image.open(path))
    realesrgan = get_realesrgan(model, device='cuda')
    images = super_resolution(images, realesrgan)
    for n in range(len(images)):
        path = paths[n].replace('.png', '')
        images[n].save(f'{path}_{model}.png')


supersize_all()
