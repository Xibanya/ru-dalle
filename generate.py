#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import pathlib
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from pynvml import *
from tqdm import tqdm
from translate import Translator

from colors import print_cyan
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.pipelines import generate_images, super_resolution

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:2048"
dev = torch.device('cuda')
torch.nn.functional.conv2d(
    torch.zeros(32, 32, 32, 32, device=dev),
    torch.zeros(32, 32, 32, 32, device=dev)
)

readable_caption = 'city at night'
input_text = Translator(to_lang='ru').translate(readable_caption)

TOP_P = 0.999
TOP_K = 2048
TEMPERATURE = 1.0
CHECKPOINT_PATH = 'checkpoints/city-night.pt'
OUTPUT_PATH = 'content/output/CityNight'
IMAGE_COUNT = 2
SUPER_RESOLUTION = True
SR = 'x2'  # x8, x4, or x2

model = get_rudalle_model('Malevich', pretrained=True, fp16=True, device='cuda')
if Path.exists(Path(CHECKPOINT_PATH)):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print(f'Loaded from {CHECKPOINT_PATH}')
vae = get_vae().to('cuda')
tokenizer = get_tokenizer()
realesrgan = get_realesrgan(SR, device='cuda') if SUPER_RESOLUTION else None

output_path = Path(OUTPUT_PATH)
output_path.mkdir(parents=True, exist_ok=True)
gc.collect()
torch.cuda.empty_cache()

for i in tqdm(range(IMAGE_COUNT), colour='green'):
    pil_images, _ = generate_images(
        text=input_text,
        tokenizer=tokenizer,
        dalle=model,
        vae=vae, top_k=TOP_K,
        images_num=1,
        top_p=TOP_P,
        temperature=TEMPERATURE
    )
    if SUPER_RESOLUTION:
        pil_images = super_resolution(pil_images, realesrgan)
    caption = readable_caption if readable_caption is not None else input_text
    sr = '_' + SR if SUPER_RESOLUTION else ''
    save_index = 0
    save_name = f'{caption}_{save_index:03d}'
    save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    while pathlib.Path.exists(save_path):
        save_index = save_index + 1
        save_name = f'{caption}_{save_index:03d}'
        save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    print_cyan(f'saving {save_path}')
    for n in range(len(pil_images)):
        pil_images[n].save(save_path)

gc.collect()
torch.cuda.empty_cache()
