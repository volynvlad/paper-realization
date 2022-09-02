from multiprocessing.sharedctypes import Value
import os
import yaml

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from cnn_sentence_classification.model import CNNSentenceClassifier
from word2vec.model import (CBOW_Model,
                            SkipGram_Model,
                            SkipGramNegativeSamplingTrainer)


def get_model_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Model
    elif model_name == "skipgram":
        return SkipGram_Model
    elif model_name == "skipgram_neg":
        return SkipGramNegativeSamplingTrainer
    elif model_name == "cnn_sentence_cl":
        return CNNSentenceClassifier
    else:
        raise ValueError("model_name should be: "
                         "cnn_sentence_cl, cbow, skipgram or skipgram_neg")


def get_optimizer_class(name: str):
    if name == "adam":
        return optim.Adam
    elif name == "adagrad":
        return optim.Adagrad
    else:
        raise ValueError("Optimizer name should be adam or adagrad")


def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.
    """
    def lr_lambda(epoch):
        return (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer,
                            lr_lambda=lr_lambda,
                            verbose=verbose)
    return lr_scheduler


def save_config(config: dict, model_dir: str):
    """Save config file to `model_dir` directory"""
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)


def save_vocab(vocab, model_dir: str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)
