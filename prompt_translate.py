#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
from translate import Translator
from colors import print_cyan, print_blue

FILE = 'prompts.csv'


def open_translations():
    translations = csv.reader(open(
        file=FILE, encoding='utf-8', errors='ignore', mode="r"), delimiter=",")
    return translations


def existing_translation(prompt: str):
    """
    If the translation has already been done, returns it, otherwise returns None
    """
    translations = open_translations()
    for row in translations:
        if prompt == row[0]:
            print_blue(f'found existing translation of "{prompt}": {row[1]}')
            return row[1]
    return None


def new_translation(prompt: str) -> str:
    """
    Retrieves a new translation from the Translator API and appends it to the dataset
    :return: machine translation of input phrase
    """
    translation = Translator(to_lang='ru').translate(prompt)
    print_cyan(f'translated "{prompt}": {translation}')
    with open(FILE, mode='a', newline='', encoding='utf-8') as t:
        writer = csv.writer(t, delimiter=',')
        writer.writerow([prompt, translation])
    return translation


def translate(prompt: str) -> str:
    existing = existing_translation(prompt)
    return existing if existing is not None else new_translation(prompt)
