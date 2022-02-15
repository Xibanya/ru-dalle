#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import datetime
import gc
import pathlib
from pathlib import Path

import torch
import torch.nn.functional as F
from pynvml import *
from tqdm import tqdm

from colors import print_cyan
from rudalle import get_rudalle_model, get_tokenizer, get_vae
from rudalle.pipelines import generate_images

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:2048"
dev = torch.device('cuda')
torch.nn.functional.conv2d(
    torch.zeros(32, 32, 32, 32, device=dev),
    torch.zeros(32, 32, 32, 32, device=dev)
)

input_text = u"город на закате"
readable_caption = 'city at sunset'
TOP_P = 0.999
TOP_K = 2048
TEMPERATURE = 1.0
CHECKPOINT_PATH = 'checkpoints/rudalle_dalle_last.pt'
OUTPUT_PATH = 'content/output'
IMAGE_COUNT = 4

model = get_rudalle_model('Malevich', pretrained=True, fp16=True, device='cuda')
if Path.exists(Path(CHECKPOINT_PATH)):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print(f'Loaded from {CHECKPOINT_PATH}')
vae = get_vae().to('cuda')
tokenizer = get_tokenizer()

output_path = Path(OUTPUT_PATH)
output_path.mkdir(parents=True, exist_ok=True)


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
    caption = readable_caption if readable_caption is not None else input_text
    save_name = f"{caption}_{str(i)}"
    save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    if pathlib.Path.exists(save_path):
        save_name = save_name + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_path = pathlib.Path(os.path.join(output_path, save_name + '.png'))
    print_cyan(f'saving {save_name}')
    for n in range(len(pil_images)):
        pil_images[n].save(save_path)

gc.collect()
torch.cuda.empty_cache()
