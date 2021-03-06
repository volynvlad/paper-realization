import argparse
import collections
from functools import wraps
from time import time
import re
import yaml

import numpy as np
from torch import reshape
from torch.utils.data import DataLoader

TOKEN_RE = re.compile(r"[\w\d]+")


def setup_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="path to yaml config")
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        print(f"{func.__name__} is running...")
        start = time()
        result = func(*args, **kw)
        end = time()
        print(f"{func.__name__} took: {end-start:2.4f} sec")
        return result

    return wrap


def tokenize_text_simple_regex(text, min_token_size=4):
    all_token = TOKEN_RE.findall(text.lower())
    return [token for token in all_token if len(token) >= min_token_size]


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **kwargs):
    return np.array([tokenizer(text, **kwargs) for text in texts],
                    dtype=object)


def build_vocabulary(tokenized_texts,
                     max_size=1000000,
                     max_doc_freq=0.9,
                     min_count=4,
                     pad_word=None):
    word_counts = collections.defaultdict(int)
    doc_n = 0

    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1

    word_counts = {
        word: c
        for word, c in word_counts.items()
        if c >= min_count and c / doc_n <= max_doc_freq
    }

    sorted_word_counts = sorted(word_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0), *sorted_word_counts]

    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    word2freq = np.array([c / doc_n for _, c in sorted_word_counts],
                         dtype="float32")

    return word2id, word2freq


def texts2token_ids(tokenized_texts, word2id):
    return np.array([np.array([word2id[token]
                     for token in text if token in word2id])
                     for text in tokenized_texts], dtype=object)


def check_accuracy(model, train_dataset, test_dataset):
    # when we used balanced dataset we can use accuracy as a score

    train_x, train_y = next(iter(DataLoader(
        train_dataset,
        batch_size=len(train_dataset)
    )))
    train_pred = model(train_x)
    train_pred = reshape(train_pred, (len(train_pred),))
    train_pred = np.array([1 if x >= 0.5 else 0 for x in train_pred],
                          dtype=np.float32)
    train_y = train_y.numpy()

    print(np.unique(train_pred, return_counts=True))
    print(np.unique(train_y, return_counts=True))
    acc = (train_pred == train_y).sum()
    print(f"test accuracy: {acc} / {len(train_pred)}"
          f" = {1.0 * acc / len(train_pred):.4f}")

    print("="*100)
    test_x, test_y = next(iter(
        DataLoader(test_dataset, batch_size=len(test_dataset))
    ))
    test_pred = model(test_x)
    test_pred = reshape(test_pred, (len(test_pred),))
    test_pred = np.array([1 if x >= 0.5 else 0 for x in test_pred],
                         dtype=np.float32)
    test_y = test_y.numpy()

    print(np.unique(test_pred, return_counts=True))
    print(np.unique(test_y, return_counts=True))
    acc = (test_pred == test_y).sum()
    print(f"test accuracy: {acc} / {len(test_pred)}"
          f" = {1.0 * acc / len(test_pred):.4f}")
