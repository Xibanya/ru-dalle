#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import glob
import pathlib
import warnings
from pathlib import Path
from random import shuffle, random

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from pynvml import *
from tqdm import tqdm

from colors import print_cyan, print_blue, print_green
from postfx import apply_to_pil
from prompt_translate import translate
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.image_prompts import ImagePrompts
from rudalle.pipelines import generate_images, super_resolution
from rudalle.utils import seed_everything

with open(f'config.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# ############ ARGS HERE ################
DEFAULT_CAPTION = "\u0420\u0438\u0447\u0430\u0440\u0434 \u0414. \u0414\u0436\u0435\u0439\u043C\u0441 Aphex Twin"
CAPTION = cfg['gen_prompt'] if cfg['gen_prompt'] else DEFAULT_CAPTION
TRANSLATE = cfg['translate']
MODEL_NAME = cfg['gen_model'] if cfg['gen_model'] else 'Malevich'
CHECKPOINT_PATH = f'checkpoints/{MODEL_NAME}.pt' if cfg['gen_model'] else None
FILE_NAME = cfg['file_name'] if cfg['file_name'] else CAPTION
OUTPUT_PATH = f'content/output/{MODEL_NAME}'
if cfg['output_dir']:
    OUTPUT_PATH = OUTPUT_PATH + '/' + cfg['output_dir']
PROMPT_PATH = f"content/Data/{MODEL_NAME}/Prompt" if cfg['use_image_prompts'] else None
TEMPERATURE = cfg['temperature']
IMAGE_COUNT = cfg['image_count']
SUPER_RESOLUTION = cfg['super_res']
SR = cfg['upscale']
INCREMENT_FROM = 0  # increment file name number from here
TOP_K = cfg['top_k']
TOP_P = cfg['top_p']
POST_FX = cfg['post_fx']
# ######################################

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
    print_blue(f'Loaded from {CHECKPOINT_PATH}')
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
topk_index = 0
prompt_index = 0

if cfg['seed']:
    seed_everything(cfg['gen_seed'])
    print_green(f"\nseed {cfg['gen_seed']}\n")

paths = []
prompt_imgs = []
if PROMPT_PATH is not None:
    types = ('*.png', '*.jpg', "*.jpeg", "*.bmp")
    for ext in types:
        paths.extend(glob.glob(os.path.join(PROMPT_PATH, ext)))
    for path in paths:
        prompt_imgs.append(Image.open(path).resize((256, 256)))

if cfg['shuffle_start']:
    shuffle(TEMPERATURE)
    shuffle(TOP_P)
    shuffle(TOP_K)
    if len(prompt_imgs) > 1:
        shuffle(prompt_imgs)

scores = {}

for i in tqdm(range(IMAGE_COUNT), colour='green'):
    temp = TEMPERATURE[temp_index]
    top_p = TOP_P[topp_index]
    top_k = TOP_K[topk_index]
    if len(prompt_imgs) > 0:
        if prompt_index > len(prompt_imgs) - 1:
            prompt_index = 0
            if cfg['shuffle_loop'] and len(prompt_imgs) > 1:
                shuffle(prompt_imgs)
        prompt_img = prompt_imgs[prompt_index]
        if cfg['prompt_flip'] > 0.0 and random() <= cfg['prompt_flip']:
            prompt_img = ImageOps.flip(prompt_img)
        prompt_index = prompt_index + 1
        image_prompt = ImagePrompts(prompt_img, borders, vae, 'cuda', crop_first=True)
        pil_images, score = generate_images(image_prompts=image_prompt, text=input_text, tokenizer=tokenizer,
                                            dalle=model, vae=vae, images_num=1, top_k=top_k, top_p=top_p,
                                            temperature=temp)
    else:
        pil_images, score = generate_images(text=input_text, tokenizer=tokenizer, dalle=model,
                                            vae=vae, images_num=1, top_k=top_k, top_p=top_p, temperature=temp)

    temp_index = temp_index + 1
    topp_index = topp_index + 1
    topk_index = topk_index + 1
    if temp_index > len(TEMPERATURE) - 1:
        temp_index = 0
        if cfg['shuffle_loop']:
            shuffle(TEMPERATURE)
    if topp_index > len(TOP_P) - 1:
        topp_index = 0
        if cfg['shuffle_loop']:
            shuffle(TOP_P)
    if topk_index > len(TOP_K) - 1:
        topk_index = 0
        if cfg['shuffle_loop']:
            shuffle(TOP_K)
    if SUPER_RESOLUTION:
        pil_images = super_resolution(pil_images, realesrgan)
    caption = CAPTION if CAPTION is not None else input_text
    save_index = INCREMENT_FROM + i
    save_prefix = f'{FILE_NAME}_s{int(score[0])}_t{temp}_p{top_p}_k{top_k}'
    save_name = f'{save_prefix}_{save_index:03d}'
    save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    while pathlib.Path.exists(save_path):
        save_index = save_index + 1
        save_name = f'{save_prefix}_{save_index:03d}'
        save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    for n in range(len(pil_images)):
        scores[save_name] = score[n]
        if not POST_FX or cfg['save_both']:
            pil_images[n].save(save_path)
        print_cyan(f'\n{save_path}, score: {int(score[n])}')
        if POST_FX:
            apply_to_pil(pil_images[n], output_path, save_name + '_fx',
                         noise=cfg['noise'],
                         noise_strength=cfg['noise_strength'],
                         clip_limit=cfg['clip_limit'],
                         sigma_a=cfg['sigma_a'],
                         sigma_b=cfg['sigma_b']
                         )

gc.collect()
torch.cuda.empty_cache()
