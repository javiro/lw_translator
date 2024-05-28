import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from io import open
import unicodedata
import re
from src.utils.language import Lang


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False, dataset="train"):
    print("Reading lines...")

    assert (lang1 in ["en", "da", "nb", "sv"]) & (lang2 in ["en", "da", "nb", "sv"])

    # Read the file and split into lines

    lines_metadata = np.array(open(f'data/{dataset}.metadata', encoding='utf-8').\
    read().strip().split('\n'))
    lines_src = np.array(open(f'data/{dataset}.src', encoding='utf-8').\
        read().strip().split('\n'))
    lines_tgt = np.array(open(f'data/{dataset}.tgt', encoding='utf-8').\
        read().strip().split('\n'))
    lines = np.where(lines_metadata=='{"src_lang": "%s", "tgt_lang": "%s"}' % (lang1, lang2))[0]
    print("Total lines: ", len(lines))
    # Split every line into pairs and normalize
    pairs = [[normalizeString(l1), normalizeString(l2)] for l1, l2 in zip(lines_src[lines], lines_tgt[lines])]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, max_length, dataset, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, dataset)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

