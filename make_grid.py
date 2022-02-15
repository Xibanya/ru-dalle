from datetime import datetime
import glob
import os

from PIL import Image
import colors

FOLDER = 'content/output/Data'

ROWS = 4
COLS = 4
SIZE = 256

types = ('*.png', '*.jpg', "*.jpeg", "*.bmp")
paths = []
for ext in types:
    paths.extend(glob.glob(os.path.join(FOLDER, ext)))

colors.print_blue(f"{len(paths)} images found")

new = Image.new("RGB", (SIZE * COLS, SIZE * ROWS))

for y in range(ROWS):
    for x in range(COLS):
        index = x + y * COLS
        if index > len(paths) - 1:
            index = index % len(paths)
        img = Image.open(paths[index])
        new.paste(img, (x * SIZE, y * SIZE))

now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_path = FOLDER + f'/grid_{ROWS}x{COLS}_{now}.png'
new.save(save_path)
colors.print_cyan(f'saved to {save_path}')
new.show()
