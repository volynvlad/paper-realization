import re
import collections
from time import time
from functools import wraps

import numpy as np

TOKEN_RE = re.compile(r'[\w\d]+')


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        print(f"{f.__name__} is running...")
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'{f.__name__} took: {te-ts:2.4f} sec')
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
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    word2freq = np.array([c / doc_n for _, c in sorted_word_counts],
                         dtype='float32')

    return word2id, word2freq


def texts2token_ids(tokenized_texts, word2id):
    return np.array([np.array([word2id[token]
                     for token in text if token in word2id])
                     for text in tokenized_texts], dtype=object)
