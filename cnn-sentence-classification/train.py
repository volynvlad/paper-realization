#!/usr/bin/python3.10
"""

train file to run word2vec and cnn models

"""
import os

import numpy as np
import torch
from spacy import load

from cnn_sentence_classifier import CNNSentenceClassifier
from data import read_data, generate_data_labels, shuffle_data_labels
from data import (PaddedSequenceDataset,
                  SkipGramNegativeSamplingTrainer,
                  Embeddings,
                  train_eval_loop,
                  no_loss)
from utils import (
    check_accuracy,
    setup_config,
    tokenize_corpus,
    build_vocabulary,
    texts2token_ids,
    timing,
)


@timing
def train_word2vec(train_dataset: PaddedSequenceDataset,
                   test_dataset: PaddedSequenceDataset,
                   vocabulary: dict,
                   config: dict):
    trainer = SkipGramNegativeSamplingTrainer(len(vocabulary),
                                              config["features_num"],
                                              config["max_sentence_len"],
                                              radius=config["radius"],
                                              negative_samples_n=32)
    if (not config["train_new_word2vec_model"]
            and os.path.isfile(config["word2vec_model_path"])):
        print("Loading word2vec model...")
        trainer.load_state_dict(torch.load(config["word2vec_model_path"]))
        return trainer

    print("Training word2vec model...")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"test dataset size: {len(test_dataset)}")

    _, best_model = train_eval_loop(
        trainer,
        train_dataset,
        test_dataset,
        no_loss,
        lr=config["word2vec_lr"],
        epoch_n=200,
        batch_size=48,
        device=config["device"],
        early_stopping_patience=10,
        max_batches_per_epoch_train=3000,
        max_batches_per_epoch_val=len(test_dataset),
        lr_scheduler_ctor=lambda optimizer: torch.optim.lr_scheduler.
        ReduceLROnPlateau(optimizer, patience=3),
        is_embedding=True)

    torch.save(best_model.state_dict(), config["word2vec_model_path"])

    return best_model


@timing
def train_cnn(train_dataset,
              test_dataset,
              config):
    cnn_model = CNNSentenceClassifier(window_size=config["window_size"],
                                      features_num=config["features_num"],
                                      sentence_length=config["max_sentence_len"],
                                      device="cpu")
    print(str(cnn_model))
    if (not config["train_new_cnn_model"] and
            os.path.isfile(config["cnn_model_path"])):
        print("Loading cnn model...")
        cnn_model.load_state_dict(torch.load(config["cnn_model_path"]))
        return cnn_model

    print("Training cnn model...")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"test dataset size: {len(test_dataset)}")

    _, best_model = train_eval_loop(
        cnn_model,
        train_dataset,
        test_dataset,
        torch.nn.BCELoss(),
        lr=config["cnn_lr"],
        epoch_n=200,
        batch_size=32,
        # device="cuda",
        device=config["device"],
        early_stopping_patience=20,
        max_batches_per_epoch_train=4000,
        max_batches_per_epoch_val=len(test_dataset),
        optimizer_ctor=torch.optim.Adagrad,
        lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.
        ReduceLROnPlateau(optim, patience=3))
    torch.save(cnn_model.state_dict(), config["cnn_model_path"])

    return best_model


def main():
    np.random.seed(373)

    config = setup_config()
    data_plot, data_quote = read_data(config)

    # get text data
    data, labels = generate_data_labels(data_plot, data_quote)
    data, labels = shuffle_data_labels(data, labels)

    train_size = int(len(data) * 0.9)

    # split text data
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    # tokenize text data
    train_tokenized = tokenize_corpus(train_data)
    test_tokenized = tokenize_corpus(test_data)

    # build vocabulary from train text data
    vocabulary, _ = build_vocabulary(train_tokenized,
                                     max_doc_freq=0.9,
                                     min_count=5,
                                     pad_word="<PAD>")
    print(f"vocabulary size = {len(vocabulary)}")

    # make ids from tokens
    train_token_ids = texts2token_ids(train_tokenized, vocabulary)
    test_token_ids = texts2token_ids(test_tokenized, vocabulary)

    # add paddings to data
    train_dataset = PaddedSequenceDataset(train_token_ids,
                                          np.zeros(len(train_token_ids)),
                                          out_len=config["max_sentence_len"])
    test_dataset = PaddedSequenceDataset(test_token_ids,
                                         np.zeros(len(test_token_ids)),
                                         out_len=config["max_sentence_len"])

    trainer = train_word2vec(train_dataset,
                             test_dataset,
                             vocabulary,
                             config)
    tagger = load(
        "en_core_web_sm",
        disable=["tok2vec",
                 "parser",
                 "senter",
                 "lemmatizer",
                 "ner",
                 ]
        )
    embeddings = Embeddings(trainer.center_emb.weight.detach().cpu().numpy(),
                            vocabulary,
                            tagger=tagger)

    train_dataset = PaddedSequenceDataset(train_token_ids,
                                          train_labels,
                                          embeddings=embeddings,
                                          out_len=config["max_sentence_len"])
    test_dataset = PaddedSequenceDataset(test_token_ids,
                                         test_labels,
                                         embeddings=embeddings,
                                         out_len=config["max_sentence_len"])

    model = train_cnn(train_dataset, test_dataset, config).eval()
    check_accuracy(model, train_dataset, test_dataset)


if __name__ == "__main__":
    main()
