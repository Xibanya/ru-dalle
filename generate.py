#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import glob
import pathlib
import warnings
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from pynvml import *
from random import shuffle
from tqdm import tqdm
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.image_prompts import ImagePrompts
from rudalle.pipelines import generate_images, super_resolution
from prompt_translate import translate
from postfx import apply_to_pil

# ############ ARGS HERE ################
DEFAULT_CAPTION = "\u0420\u0438\u0447\u0430\u0440\u0434 \u0414. \u0414\u0436\u0435\u0439\u043C\u0441 Aphex Twin"
CAPTION = 'illustration of a city in a veil of fog'
TRANSLATE = True
MODEL_NAME = 'City'
CHECKPOINT_PATH = f'checkpoints/{MODEL_NAME}.pt'  # put None to use Malevich XL straight-up
OUTPUT_PATH = f'content/output/{MODEL_NAME}'
FILE_NAME = f'{MODEL_NAME}_{CAPTION}'
PROMPT_PATH = None  # f'content/Data/{MODEL_NAME}/Prompt'
TOP_P = [0.999]
TOP_K = 2048
TEMPERATURE = [1.05, 1.0, 0.95, 0.9, 1.1]
SHUFFLE_START = True  # shuffle initial lists
SHUFFLE_ON_LOOP = True  # if looping around, shuffle lists
IMAGE_COUNT = 4
SUPER_RESOLUTION = True
SR = 'x4'  # x8, x4, or x2
INCREMENT_FROM = 0
POST_FX = True
# ######################################


def print_cyan(msg):
    print('\033[96m' + msg + '\033[96m')


warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:2048"
dev = torch.device('cuda')
torch.nn.functional.conv2d(
    torch.zeros(32, 32, 32, 32, device=dev),
    torch.zeros(32, 32, 32, 32, device=dev)
)
gc.collect()
torch.cuda.empty_cache()

model = get_rudalle_model('Malevich', pretrained=True, fp16=True, device='cuda')
if CHECKPOINT_PATH is not None and Path.exists(Path(CHECKPOINT_PATH)):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print(f'Loaded from {CHECKPOINT_PATH}')
vae = get_vae().to('cuda')
tokenizer = get_tokenizer()
realesrgan = get_realesrgan(SR, device='cuda') if SUPER_RESOLUTION else None

output_path = Path(OUTPUT_PATH)
output_path.mkdir(parents=True, exist_ok=True)

input_text = CAPTION

if TRANSLATE and CAPTION is not DEFAULT_CAPTION:
    input_text = translate(CAPTION)
else:
    print_cyan(f'prompt: {CAPTION}')

borders = {'up': 4, 'left': 0, 'right': 0, 'down': 0}
temp_index = 0
topp_index = 0
prompt_index = 0

paths = []
prompt_imgs = []
if PROMPT_PATH is not None:
    types = ('*.png', '*.jpg', "*.jpeg", "*.bmp")
    for ext in types:
        paths.extend(glob.glob(os.path.join(PROMPT_PATH, ext)))
    for path in paths:
        prompt_imgs.append(Image.open(path).resize((256, 256)))

if SHUFFLE_START:
    shuffle(TEMPERATURE)
    shuffle(TOP_P)
    if len(prompt_imgs) > 1:
        shuffle(prompt_imgs)

for i in tqdm(range(IMAGE_COUNT), colour='green'):
    temp = TEMPERATURE[temp_index]
    top_p = TOP_P[topp_index]
    if len(prompt_imgs) > 0:
        if prompt_index > len(prompt_imgs) - 1:
            prompt_index = 0
            if SHUFFLE_ON_LOOP and len(prompt_imgs) > 1:
                shuffle(prompt_imgs)
        prompt_img = prompt_imgs[prompt_index]
        prompt_index = prompt_index + 1

        image_prompt = ImagePrompts(prompt_img, borders, vae, 'cuda', crop_first=True)
        pil_images, _ = generate_images(image_prompts=image_prompt, text=input_text, tokenizer=tokenizer, dalle=model,
                                        vae=vae, images_num=1, top_k=TOP_K, top_p=top_p, temperature=temp)
    else:
        pil_images, _ = generate_images(text=input_text, tokenizer=tokenizer, dalle=model,
                                        vae=vae, images_num=1, top_k=TOP_K, top_p=top_p, temperature=temp)
    temp_index = temp_index + 1
    topp_index = topp_index + 1
    if temp_index > len(TEMPERATURE) - 1:
        temp_index = 0
        if SHUFFLE_ON_LOOP and len(TEMPERATURE) > 1:
            shuffle(TEMPERATURE)
    if topp_index > len(TOP_P) - 1:
        topp_index = 0
        if SHUFFLE_ON_LOOP and len(TOP_P) > 1:
            shuffle(TOP_P)
    if SUPER_RESOLUTION:
        pil_images = super_resolution(pil_images, realesrgan)
    caption = CAPTION if CAPTION is not None else input_text
    sr = '_' + SR if SUPER_RESOLUTION else ''
    save_index = INCREMENT_FROM + i
    save_prefix = f'{FILE_NAME}_t{temp}'
    save_name = f'{save_prefix}_{save_index:03d}'
    save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    while pathlib.Path.exists(save_path):
        save_index = save_index + 1
        save_name = f'{save_prefix}_{save_index:03d}'
        save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    print_cyan(f'saving {save_path}')
    for n in range(len(pil_images)):
        pil_images[n].save(save_path)
        if POST_FX:
            apply_to_pil(pil_images[n], output_path, save_name + '_fx')

gc.collect()
torch.cuda.empty_cache()
