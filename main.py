#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
NOTE FROM XIBANYA:
This was an attempt to get Looking Glass v1.1
https://colab.research.google.com/drive/11vdS9dpcZz2Q2efkOjcwyax4oob6N40G?usp=sharing
working locally. This code is mostly the cells from the colab notebook pasted in and
cleaned up, with some changes to solve exceptions I was receiving locally that
didn't occur in the notebook.

note most of the Looking Glass v1.1 notebook is derived from this notebook from the dalle repo:
https://colab.research.google.com/drive/1Tb7J4PvvegWOybPfUubl5O7m5I24CBg5?usp=sharing#scrollTo=g2j_g_T7wiQd
"""
import csv
import datetime
import gc
import glob
import math
import multiprocessing
import os.path
import pathlib
import random
import warnings
from functools import partial

import PIL
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import transformers
from PIL import Image
from einops import rearrange
from psutil import virtual_memory
from pynvml import *
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler, adam
from tqdm import tqdm
from transformers import AdamW
import yaml
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle import utils
from rudalle.dalle.utils import exists, is_empty
from rudalle.pipelines import generate_images, show, super_resolution

from colors import print_cyan, print_blue, print_green, print_warn
from prompt_translate import translate

with open(f'config.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

#####################################################
"""
XIB NOTE: if true, looks for a checkpoint in directory 
'checkpoints/{RESUME_FROM}.pt'
"""
RESUME = cfg['resume']
MODEL_NAME = cfg['train_model']
RESUME_FROM = MODEL_NAME
SAVE_PATH = 'checkpoints/'
OUTPUT_FOLDER = MODEL_NAME
INPUT_FOLDER = 'Data/' + MODEL_NAME
INDIVIDUAL_DESCRIPTION_FILE = 'data_desc.csv'  # leave as 'data_desc.csv' if none

""" 
XIB NOTE: probably use DEFAULT_CAPTION when TRANSLATE is False?
"""
DEFAULT_CAPTION = "\u0420\u0438\u0447\u0430\u0440\u0434 \u0414. \u0414\u0436\u0435\u0439\u043C\u0441 Aphex Twin"
CAPTION = cfg['train_prompt'] if cfg['train_prompt'] else DEFAULT_CAPTION
SAVE_NAME = CAPTION  # file name prefix for the saved images
# set to false if all the translation API calls for the day have been used up or input is in Russian already
TRANSLATE = cfg['translate']

""" XIB NOTE: these are constants, don't change these! """
LOW = "Low"  # 1e-5
MEDIUM = "Medium"  # 2e-5
HIGH = "High"  # 1e-4
CUSTOM = "Custom"
"""
Universe similarity determines how close to the original images you will receive. 
High similarity produces alternate versions of an image, low similarity produces 
"variations on a theme".
"""
CUSTOM_LEARNING_RATE = float(cfg['custom_lr'])
universe_similarity = MEDIUM
width = 256
height = 256
BATCH_SIZE = 1
"""
The amount of epochs that training occurs for. 
Turn down if the images are too similar to the base image. 
Turn up if they're too different. Use this for fine adjustments.
"""
EPOCH_AMOUNT = cfg['epochs']
WARMUP_STEPS = cfg['warmup_steps']
"""
XIB NOTE: this prints epoch number when iterating over epochs
set to <= 0 to not use
"""
LOG_EPOCH = cfg['log_epoch']
"""
XIB NOTE: Generates an image every PREVIEW_EPOCH epochs
set to <= 0 to not use
"""
PREVIEW_EPOCH = cfg['preview_epoch']
"""
XIB NOTE: saves checkpoint every SAVE_EVERY epochs (saves checkpoint at end regardless)
set to <= 0 to not use
"""
SAVE_EPOCH = cfg['save_epoch']
"""
XIB NOTE: put this in a /content directory
Only used if multiple_image_tuning is False
"""
INPUT_IMAGE = "testpic.png"
"""
Multiple Image Tuning
Set this variable to "True" to enable Multiple Image Tuning.
This attempts to make a collage trained on every image in the folder at once.
"""
multiple_image_tuning = True

"""
If the folder has no images or `folder_to_train` is `""`, Multiple Image Tuning will be skipped
even if multiple_image_tuning is True
"""
folder_to_train = INPUT_FOLDER

"""
if true has a visualization of the training loss pop up; 
execution of the script halts until you dissmiss it.
"""
SHOW_TRAINING_PLOTS = False
"""
Input text can influence the end result you get to a minor degree, so you have the 
option to change it now.
THIS MUST HAVE AT LEAST ONE CHARACTER IN IT IN IT OTHERWISE THE FINE-TUNING WILL BREAK.
Input text **must be in Russian**. You do not need to change this, though, 
unless you are a perfectionist.

XIB NOTE: this is the original prompt from bearsharktopus's notebook but to use it here
you have to encode/decode to utf8. the provided prompt is the same thing but in latin script
"\u0420\u0438\u0447\u0430\u0440\u0434 \u0414. \u0414\u0436\u0435\u0439\u043C\u0441 Aphex Twin"
#.encode('utf8') to get around unicode errors?
"""
# u"город" (city)
# u prefix is needed for cyrillic to work
input_text = CAPTION
if TRANSLATE and CAPTION is not DEFAULT_CAPTION:
    input_text = translate(CAPTION)
else:
    print_cyan(f'prompt: {CAPTION}')


"""
If you *really* want to make a 9 or 25 image collage but have a weak CPU, you can try turning on low mem mode. 
It will take a *while* though.
"""
low_mem = False

"""
By default, Looking Glass includes your original image(s) somewhere in the collage as "Ground Truth". 
Check this box to disable that behavior.
"""
skip_gt = True

"""
If you'd like to change the shape or size of the output from its default 256x256 set "resize" to true.
Note that this is **much slower**. Not only is the process itself slower but it forces itself to run with a 
batch_size of 1, meaning it forces you into low_mem mode, which makes pictures take a while.
"""
resize = False

"""
XIB NOTE: if you do ML generation a lot you already know what these are
"""
TOP_P = 0.999
TOP_K = 2048
TEMPERATURE = 1.0

"""
XIB NOTE: switch these up to get different kinds of output. fun!
"""
ADAMW = 'AdamW'
ADAM = 'Adam'
OPTIMIZER = ADAMW

"""
XIB NOTE: DON'T CHANGE THESE
"""
ONE_IMAGE = "1 image"
FOUR_IMAGES = "4 images"
NINE_IMAGES = "9 images"
TWENTY_FIVE_IMAGES = "25 images"
"""
XIB NOTE: the above four "constants" are the only valid values for IMAGE_AMOUNT
"""
IMAGE_AMOUNT = ONE_IMAGE

""" To generate images use generate.py instead """
TRAIN = True
##############################################################

# XIB NOTE: these shenanigans to prevent cuda not initialized errors
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:2048"
dev = torch.device('cuda')
torch.nn.functional.conv2d(
    torch.zeros(32, 32, 32, 32, device=dev),
    torch.zeros(32, 32, 32, 32, device=dev)
)

ram_gb = round(virtual_memory().total / 1024 ** 3, 1)

print_cyan(f'CPU: {multiprocessing.cpu_count()}')
print_cyan(f'RAM GB: {ram_gb}')
print_cyan(f"PyTorch version: {torch.__version__}")
print_cyan(f"CUDA version: {torch.version.cuda}")
print_cyan(f"cuDNN version: {torch.backends.cudnn.version()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_cyan(f"device: {device.type}")
warnings.filterwarnings("ignore", category=UserWarning)

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
if info.total > 16252636672:
    print_green('Everything is ok, you can begin')
else:
    print_warn('Potential OOM issues?')

device = 'cuda'
model = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
if RESUME:
    model_path = os.path.join(SAVE_PATH, f"{RESUME_FROM}.pt")
    model.load_state_dict(torch.load(model_path))
    print_blue(f'Loaded pretrained model from {model_path}')
else:
    print_blue('Loaded default Malevich model')

vae = get_vae().to(device)
tokenizer = get_tokenizer()

file_to_train = INPUT_IMAGE
epoch_amt = EPOCH_AMOUNT

if universe_similarity == HIGH:
    learning_rate = 1e-4
elif universe_similarity == MEDIUM:
    learning_rate = 2e-5
elif universe_similarity == LOW:
    learning_rate = 1e-5
else:
    learning_rate = CUSTOM_LEARNING_RATE

if TRAIN:
    print_cyan(f'Universe similarity: {universe_similarity}, learning rate: {learning_rate}')


def save_images(image_data, suffix: str):
    for j in range(len(image_data)):
        s_folder = 'content/output/' + folder_to_train + '/'
        pathlib.Path(s_folder).mkdir(parents=True, exist_ok=True)
        s_name = f"{folder_to_train}_{CAPTION}_{str(n)}{suffix}"
        s_path = pathlib.Path(os.path.join(s_folder, s_name + '.png'))
        if pathlib.Path.exists(s_path):
            s_name = s_name + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        print_cyan(f'attempting to save {s_name}')
        pil_images[n].save(s_folder + s_name + '.png')


class Args:
    def __init__(self):
        self.text_seq_length = model.get_param('text_seq_length')
        self.total_seq_length = model.get_param('total_seq_length')
        self.epochs = epoch_amt
        self.save_path = SAVE_PATH
        self.model_name = MODEL_NAME
        self.save_every = SAVE_EPOCH
        self.prefix_length = 10
        self.bs = BATCH_SIZE
        self.clip = 0.25
        self.lr = learning_rate
        self.warmup_steps = WARMUP_STEPS
        self.wandb = False


args = Args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

image_amount = IMAGE_AMOUNT
collage_amount = 1

low_mem = True if resize else False

w = round(width / 8)
h = round(height / 8)

files_to_train = []
types = ('*.png', '*.jpg', "*.jpeg", "*.bmp")

if multiple_image_tuning:
    for ext in types:
        files_to_train.extend(glob.glob(os.path.join('content/' + folder_to_train, ext)))
    if TRAIN:
        print_blue(f"{len(files_to_train)} training images found")

if folder_to_train == "" or len(files_to_train) == 0:
    multiple_image_tuning = False

if not multiple_image_tuning:
    folder_to_train = ""

with open('data_desc.csv', 'w', newline='', encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['', 'name', 'caption'])
    spamwriter.writerow(['0', file_to_train, input_text])
file_to_train = "/content/" + file_to_train

if multiple_image_tuning:
    data_folder = f'content/{INPUT_FOLDER}'
    data_file_path = pathlib.Path(os.path.join(data_folder, INDIVIDUAL_DESCRIPTION_FILE))
    if pathlib.Path.exists(data_file_path):
        with open(data_file_path, mode='r') as infile:
            reader = csv.reader(infile)
            desc_dict = {rows[0]: translate(rows[1]) for rows in reader}
    else:
        desc_dict = {'file': 'desc'}
    with open('data_desc.csv', 'w', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['', 'name', 'caption'])
        for i in range(len(files_to_train)):
            files_to_train[i] = files_to_train[i].replace("content/" + folder_to_train + "\\", "")
            if files_to_train[i] in desc_dict.keys():
                spamwriter.writerow([i, files_to_train[i], desc_dict[files_to_train[i]]])
            else:
                spamwriter.writerow([i, files_to_train[i], input_text])


def load_image(file_path, img_name):
    image = PIL.Image.open(f'{file_path}/{img_name}')
    return image


class RuDalleDataset(Dataset):
    clip_filter_thr = 0.24

    def __init__(
            self,
            file_path,
            csv_path,
            t_tokenizer,
            resize_ratio=0.75,
            shuffle=True,
            load_first=None,
            caption_score_thr=0.6
    ):
        """ tokenizer - объект с методами tokenizer_wrapper.BaseTokenizerWrapper """

        self.text_seq_length = model.get_param('text_seq_length')
        self.tokenizer = t_tokenizer
        self.target_image_size = 256
        self.image_size = 256
        self.samples = []

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.image_size,
                                scale=(1., 1.),  # в train было scale=(0.75., 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
            # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        df = pd.read_csv(csv_path)
        # TODO: not this - Xib
        if not multiple_image_tuning:
            print_blue(f'appending {INPUT_IMAGE} to dataset')
            self.samples.append(["content", INPUT_IMAGE, input_text])
        else:
            paths = []
            for ex in types:
                paths.extend(glob.glob(os.path.join("content/" + folder_to_train, ex)))
            for path in paths:
                path = path.replace('content/', '')
                self.samples.append(["content", path, input_text])
        for caption, f_path in zip(df['caption'], df['name']):
            if 1 < len(caption) < 100 and os.path.isfile(f'{file_path}/{f_path}'):
                self.samples.append([file_path, f_path, caption])
        if shuffle:
            np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        item = item % len(self.samples)  # infinite loop, modulo dataset size
        file_path, img_name, t_text = self.samples[item]
        try:
            image = load_image(file_path, img_name)
            image = self.image_transform(image).to(device)
        except Exception as err:  # noqa
            print(err)
            random_item = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_item)
        t_text = self.tokenizer.encode_text(t_text, text_seq_length=self.text_seq_length).squeeze(0).to(device)
        return t_text, image


st = RuDalleDataset(file_path='/content/' + folder_to_train, csv_path='data_desc.csv', t_tokenizer=tokenizer)

train_dataloader = DataLoader(
    st,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

model.train()
if OPTIMIZER == ADAMW:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
elif OPTIMIZER == ADAM:
    optimizer = adam.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = AdamW(model.parameters(), lr=learning_rate)

scheduler = lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=learning_rate,
    final_div_factor=500,
    steps_per_epoch=math.ceil(len(st.samples) / BATCH_SIZE),
    epochs=epoch_amt,
    verbose=True
)


def freeze(
        freeze_emb=True,
        freeze_ln=False,
        freeze_attn=False,
        freeze_ff=True,
        freeze_other=True,
):
    for name, p in model.module.named_parameters():
        name = name.lower()
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'embeddings' in name:
            p.requires_grad = not freeze_emb
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
    return model


def train(t_model, argz: Args, t_dataloader: RuDalleDataset):
    """
    args - arguments for training
    train_dataloader - RuDalleDataset class with text - image pair in batch
    """
    loss_logs = []
    try:
        progress = tqdm(total=len(t_dataloader) * argz.epochs, desc='finetuning')
        save_counter = 0
        for epoch in range(argz.epochs):

            if LOG_EPOCH > 0 and (epoch + 1) % LOG_EPOCH == 0:
                print_blue(f"\nEpoch {epoch + 1}")

            for data_text, images in train_dataloader:
                save_counter += 1
                model.zero_grad()
                devi = t_model.get_param('device')
                attention_mask = torch.tril(
                    torch.ones((argz.bs,
                                1,
                                argz.total_seq_length,
                                argz.total_seq_length),
                               device=devi))
                image_input_ids = vae.get_codebook_indices(images)

                input_ids = torch.cat((data_text, image_input_ids), dim=1)
                _, loss = forward(model.module, input_ids, attention_mask.half(),
                                  return_loss=True, use_cache=False, gradient_checkpointing=True)
                loss = loss["image"]
                # train step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), argz.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss_logs += [loss.item()]
                progress.update()
                progress.set_postfix({"loss": loss.item()})

            if SAVE_EPOCH > 0 and (epoch + 1) % SAVE_EPOCH == 0:
                print_green(f'Saving checkpoint here {argz.model_name}_{save_counter}.pt')
                torch.save(
                    model.state_dict(),
                    os.path.join(argz.save_path, f"{argz.model_name}_{save_counter}.pt")
                )
                if SHOW_TRAINING_PLOTS:
                    plt.plot(loss_logs)
                    plt.show()
            if PREVIEW_EPOCH > 0 and (epoch + 1) % PREVIEW_EPOCH == 0:
                preview_images, _ = generate_images(text=input_text, tokenizer=tokenizer, dalle=model,
                                                    vae=vae, images_num=1, top_k=2048, top_p=0.999)
                preview_index = 0
                preview_name = f'{MODEL_NAME}_epoch{epoch + 1}_{preview_index:03d}'
                out_folder = f'content/output/{OUTPUT_FOLDER}'
                preview_save_path = pathlib.Path(os.path.join(out_folder, preview_name + '.png'))

                for p in range(len(preview_images)):
                    while pathlib.Path.exists(preview_save_path):
                        preview_index = preview_index + 1
                        preview_name = f'{MODEL_NAME}_epoch{epoch + 1}_{preview_index:03d}'
                        preview_save_path = pathlib.Path(os.path.join(out_folder, preview_name + '.png'))
                    preview_images[p].save(preview_save_path)
                    print_blue(f"\nSaved preview image to {preview_save_path}")

        if SHOW_TRAINING_PLOTS and EPOCH_AMOUNT % argz.save_every != 0:
            plt.plot(loss_logs)
            plt.show()

        torch.save(
            model.state_dict(),
            os.path.join(argz.save_path, f"{argz.model_name}.pt")
        )
        print_green(f'\nTuned and saved here: {argz.model_name}.pt')

    except KeyboardInterrupt:
        print_warn(f'What for did you stopped? Please change model_path '
                   f'to /{argz.save_path}{argz.model_name}_Failed_train.pt')
        plt.plot(loss_logs)
        plt.show()

        torch.save(
            model.state_dict(),
            os.path.join(argz.save_path, f"{argz.model_name}_Failed_train.pt")
        )
    except Exception as err:
        print_warn(f'Failed with {err}')


# idk why but this is necessary
class Layer(torch.nn.Module):
    def __init__(self, x, f, *argz, **kwargs):
        super(Layer, self).__init__()
        self.x = x
        self.f = f
        self.args = argz
        self.kwargs = kwargs

    def forward(self, x):
        return self.f(self.x(x, *self.args, **self.kwargs))


def forward(
        self,
        input_ids,
        attention_mask,
        return_loss=False,
        use_cache=False,
        gradient_checkpointing=False
):
    text = input_ids[:, :self.text_seq_length]
    text_range = torch.arange(self.text_seq_length)
    text_range += (self.vocab_size - self.text_seq_length)
    text_range = text_range.to(self.device)
    text = torch.where(text == 0, text_range, text)
    # some hardcode :)
    text = F.pad(text, (1, 0), value=2)
    text_embeddings = self.text_embeddings(text) + \
                      self.text_pos_embeddings(torch.arange(text.shape[1], device=self.device))

    image_input_ids = input_ids[:, self.text_seq_length:]

    if exists(image_input_ids) and not is_empty(image_input_ids):
        image_embeddings = self.image_embeddings(image_input_ids) + \
                           self.get_image_pos_embeddings(image_input_ids, past_length=0)
        embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
    else:
        embeddings = text_embeddings
    # some hardcode :)
    if embeddings.shape[1] > self.total_seq_length:
        embeddings = embeddings[:, :-1]

    alpha = 0.1
    embeddings = embeddings * alpha + embeddings.detach() * (1 - alpha)

    attention_mask = attention_mask[:, :, :embeddings.shape[1], :embeddings.shape[1]]
    t = self.transformer
    layers = []
    layernorms = []
    if not layernorms:
        norm_every = 0
    else:
        norm_every = len(t.layers) // len(layernorms)
    for i in range(len(t.layers)):
        layers.append(Layer(t.layers[i],
                            lambda x:
                            x[0] * layernorms[i // norm_every][0] +
                            layernorms[i // norm_every][1] if norm_every and i % norm_every == 0 else x[0],
                            torch.mul(attention_mask,
                                      t._get_layer_mask(i)[:attention_mask.size(2), :attention_mask.size(3), ]),
                            use_cache=False,
                            has_cache=False)
                      )
    if gradient_checkpointing:  # don't use this under any circumstances
        # actually please do
        # i just spent 3 hours debugging this
        embeddings = torch.utils.checkpoint.checkpoint_sequential(layers, 6, embeddings)
        transformer_output = embeddings
        present_has_cache = False
    else:
        hidden_states = embeddings
        for i in range(len(t.layers)):
            mask = torch.mul(attention_mask, t._get_layer_mask(i)[:attention_mask.size(2), :attention_mask.size(3)])
            hidden_states, present_has_cache = t.layers[i](hidden_states, mask, use_cache=use_cache)
        transformer_output = hidden_states
    transformer_output = self.transformer.final_layernorm(transformer_output)

    logits = self.to_logits(transformer_output)
    if return_loss is False:
        return logits, present_has_cache

    labels = torch.cat((text[:, 1:], image_input_ids), dim=1).contiguous().long()
    logits = rearrange(logits, 'b n c -> b c n')

    text_logits = logits[:, :self.vocab_size, :self.text_seq_length].contiguous().float()
    image_logits = logits[:, self.vocab_size:, self.text_seq_length:].contiguous().float()

    loss_text = F.cross_entropy(
        text_logits,
        labels[:, :self.text_seq_length])
    loss_img = F.cross_entropy(
        image_logits,
        labels[:, self.text_seq_length:])

    loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
    return loss, {'text': loss_text.data.detach().float(), 'image': loss_img}


# @title
model = freeze(freeze_emb=False,
               freeze_ln=False,
               freeze_attn=True,
               freeze_ff=True,
               freeze_other=False)  # freeze params to

if TRAIN:
    train(model, args, t_dataloader=st)

gc.collect()
torch.cuda.empty_cache()
vae = get_vae().to(device)


def slow_decode(self, img_seq):
    b, n = img_seq.shape
    one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=self.num_tokens).float()
    z = (one_hot_indices @ self.model.quantize.embed.weight)
    z = rearrange(z, 'b (h w) c -> b c h w', h=h
                  # int(sqrt(n))
                  )
    img = self.model.decode(z)
    img = (img.clamp(-1., 1.) + 1) * 0.5
    return img


if resize:
    vae.slow_decode = partial(slow_decode, vae)


def slow_generate_images(text: str, tokenizer, dalle, vae, top_k, top_p, images_num,
                         image_prompts=None,
                         temperature=1.0,
                         bs=8,
                         seed=None, use_cache=True, w=32, h=48):
    """
    New image generation function for arbitrary resolution from https://twitter.com/apeoffire
    """
    if seed is not None:
        utils.seed_everything(seed)
    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')
    real = 32

    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    pil_images, scores = [], []
    cache = None
    past_cache = None
    try:
        for chunk in more_itertools.chunked(range(images_num), bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = torch.tril(
                    torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
                out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
                grid = torch.zeros((h, w)).long().cuda()
                has_cache = False
                sample_scores = []
                if image_prompts is not None:
                    prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                    prompts = prompts.repeat(chunk_bs, 1)
                for idx in tqdm(range(out.shape[1], total_seq_length - real * real + w * h)):
                    idx -= text_seq_length
                    if image_prompts is not None and idx in prompts_idx:
                        out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                    else:
                        y = idx // w
                        x = idx % w
                        x_from = max(0, min(w - real, x - real // 2))
                        y_from = max(0, y - real // 2)
                        outs = []
                        xs = []
                        for row in range(y_from, y):
                            for col in range(x_from, x_from + real):
                                outs.append(grid[row, col].item())
                                xs.append((row, col))
                        for col in range(x_from, x):
                            outs.append(grid[y, col].item())
                            xs.append((y, col))
                        rev_xs = {v: k for k, v in enumerate(xs)}
                        if past_cache is not None:
                            cache = list(map(list, cache.values()))
                            rev_past = {v: k for k, v in enumerate(past_cache)}
                            for i, e in enumerate(cache):
                                for j, c in enumerate(e):
                                    t = cache[i][j]
                                    t, c = t[..., :text_seq_length, :], t[..., text_seq_length:, :]
                                    cache[i][j] = t
                            cache = dict(zip(range(len(cache)), cache))
                        past_cache = xs
                        logits, cache = dalle(torch.cat((input_ids.to(device).ravel(),
                                                         torch.from_numpy(np.asarray(outs)).long().to(device)),
                                                        dim=0).unsqueeze(0), attention_mask,
                                              cache=cache, use_cache=True, return_loss=False)
                        logits = logits[:, :, vocab_size:].view((-1, logits.shape[-1] - vocab_size))
                        logits /= temperature
                        filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                        probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                        sample = torch.multinomial(probs, 1)
                        sample_scores.append(probs[torch.arange(probs.size(0)), sample.transpose(0, 1)])
                        sample, xs = sample[-1:], xs[-1:]
                        grid[y, x] = sample.item()
                codebooks = grid.reshape((1, -1))
                images = vae.slow_decode(codebooks)
                pil_images += utils.torch_tensors_to_pil_list(images)
    except Exception as e:
        print(e)
        pass
    except KeyboardInterrupt:
        pass
    return pil_images, scores


def aspect_crop(image_path, desired_aspect_ratio):
    """
    Return a PIL Image object cropped to desired aspect ratio
    :param str image_path: Path to the image to crop
    :param str desired_aspect_ratio: desired aspect ratio in width:height format
    """
    # compute original aspect ratio
    image = Image.open(image_path)
    width, height = image.size
    original_aspect = float(width) / float(height)

    # convert string aspect ratio into float
    w, h = map(lambda x: float(x), desired_aspect_ratio.split(':'))
    computed_aspect_ratio = w / h
    inverse_aspect_ratio = h / w

    if original_aspect < computed_aspect_ratio:
        # keep original width and change height
        new_height = math.floor(width * inverse_aspect_ratio)
        height_change = math.floor((height - new_height) / 2)
        new_image = image.crop((0, height_change, width, height - height_change))
        return new_image
    elif original_aspect > computed_aspect_ratio:
        # keep original height and change width
        new_width = math.floor(height * computed_aspect_ratio)
        width_change = math.floor((width - new_width) / 2)
        new_image = image.crop((width_change, 0, width - width_change, height))
        return new_image
    elif original_aspect == computed_aspect_ratio:
        return image


"""
The output will be saved in the session structure under content/output/
"""
pil_images = []
scores = []
text = input_text
repeat = 1
rows = 2
insert = 0
amt = 4 if skip_gt else 3
if low_mem:
    repeat = 4 if skip_gt else 3
    amt = 1
if image_amount == NINE_IMAGES:
    repeat = 2
    rows = 3
    insert = 4
    amt = 4
    if low_mem:
        repeat = 8
        amt = 1
elif image_amount == TWENTY_FIVE_IMAGES:
    repeat = 6
    rows = 5
    insert = 12
    amt = 4
    if low_mem:
        repeat = 24
        amt = 1
elif image_amount == ONE_IMAGE:
    repeat = 1
    rows = 1
    amt = 1
    skip_gt = True
    insert = 0


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


for i in range(collage_amount):
    for n in range(repeat):
        for top_k, top_p, images_num in [(TOP_K, TOP_P, amt)]:
            if resize:
                _pil_images, _ = slow_generate_images(
                    text, tokenizer, model, vae, top_k=top_k,
                    images_num=images_num, top_p=top_p, w=w, h=h
                )
                pil_images += _pil_images
            else:
                _pil_images, _ = generate_images(
                    text=text,
                    tokenizer=tokenizer,
                    dalle=model,
                    vae=vae, top_k=top_k,
                    images_num=images_num,
                    top_p=top_p
                )
                pil_images += _pil_images
    if skip_gt and image_amount != "4 images":
        for top_k, top_p, images_num in [(TOP_K, TOP_P, 1)]:
            if resize:
                _pil_images, _ = slow_generate_images(
                    text, tokenizer, model, vae, top_k=top_k,
                    images_num=images_num, top_p=top_p, w=w, h=h
                )
                pil_images += _pil_images
            else:
                _pil_images, _ = generate_images(
                    text,
                    tokenizer, model, vae, top_k=top_k,
                    images_num=images_num,
                    top_p=top_p
                )
                pil_images += _pil_images

    if multiple_image_tuning:
        realesrgan = get_realesrgan('x4', device='cuda')
        pil_images = super_resolution(pil_images, realesrgan)
        file_to_train = folder_to_train + "/" + random.choice(files_to_train)
        for n in range(len(pil_images)):
            save_folder = 'content/output/' + OUTPUT_FOLDER + '/'
            pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
            save_index = 0
            save_name = f'{SAVE_NAME}_Train_{save_index:03d}'
            save_path = pathlib.Path(os.path.join(save_folder, save_name + '.png'))
            while pathlib.Path.exists(save_path):
                save_index = save_index + 1
                save_name = f'{SAVE_NAME}_Train_{save_index:03d}'
                save_path = pathlib.Path(os.path.join(save_folder, save_name + '.png'))
            print_cyan(save_path)
            pil_images[n].save(save_folder + save_name + '.png')
    else:
        ft = file_to_train.split('/')[-1]
        for n in range(len(pil_images)):
            pil_images[n].save(f"content/output/{SAVE_NAME}_Train_{str(n)}_{ft}")

    if not multiple_image_tuning and not skip_gt:
        img_path = f"content\\{INPUT_IMAGE}"
        if not resize:
            with Image.open(img_path) as im:
                # Provide the target width and height of the image
                to_insert = crop_max_square(im).resize((256, 256), Image.LANCZOS)
                to_insert = to_insert.convert('RGB')
            pil_images.insert(insert, to_insert)
        else:
            with Image.open(img_path) as im:
                # Provide the target width and height of the image
                to_insert = aspect_crop(im, (w / h)).resize((w * 8, h * 8), Image.LANCZOS)
                to_insert = to_insert.convert('RGB')
            pil_images.insert(insert, to_insert)
    show([pil_image for pil_image in pil_images], rows)

    pil_images = []
    scores = []

torch.cuda.empty_cache()
gc.collect()
