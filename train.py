#!/usr/bin/python3.10
"""

train file to run word2vec and cnn models

"""
from datetime import datetime
import os

import numpy as np
from torch import device as torch_device
from torch import load as load_torch_model
from torch import save as save_torch_model
from torch.nn import BCELoss, CrossEntropyLoss
from spacy import load as load_spacy

from cnn_sentence_classification.model import CNNSentenceClassifier
from utils.data import (
    read_data,
    generate_data_labels,
    shuffle_data_labels,
    Embeddings,
    PaddedSequenceDataset,
)
from utils.helper import (
    get_lr_scheduler,
    get_model_class,
    get_optimizer_class,
    save_config,
    save_vocab,
)
from utils.trainer import Trainer
from utils.utils import (
    RandomGenerator,
    build_vocabulary,
    check_accuracy,
    init_random_seed,
    setup_config,
    tokenize_corpus,
    texts2token_ids,
    # timing,
)

from word2vec.model import SkipGramNegativeSamplingTrainer, no_loss


# @timing
def train_word2vec(train_dataset: PaddedSequenceDataset,
                   test_dataset: PaddedSequenceDataset,
                   vocabulary: dict,
                   config: dict):
    trainer = SkipGramNegativeSamplingTrainer(len(vocabulary),
                                              config["features_num"],
                                              config["max_sentence_len"],
                                              radius=config["radius"],
                                              negative_samples_n=config["negative_samples_n"])
    if (not config["train_new_word2vec_model"]
            and os.path.isfile(config["word2vec_model_path"])):
        print("Loading word2vec model...")
        trainer.load_state_dict(load_torch_model(config["word2vec_model_path"]))
        trainer.eval()
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
        epoch_n=config["word2vec_epoch_n"],
        batch_size=config["word2vec_batch_size"],
        device=config["device"],
        early_stopping_patience=config["word2vec_early_stopping_patience"],
        max_batches_per_epoch_train=config["word2vec_max_batches_per_epoch_train"],
        max_batches_per_epoch_val=len(test_dataset),
        # lr_scheduler_ctor=lambda optimizer: lr_scheduler.
        # ReduceLROnPlateau(optimizer, patience=3),
        is_embedding=True,
        dataloader_workers_n=0,
    )

    save_torch_model(best_model.state_dict(), config["word2vec_model_path"])

    best_model.eval()

    return best_model


# @timing
def train_cnn(train_dataset,
              val_dataset,
              config):
    model_class = get_model_class(config["cnn_model_name"])
    cnn_model = model_class(window_size=config["window_size"],
                            features_num=config["features_num"],
                            sentence_length=config["max_sentence_len"],
                            device="cpu")
    print(str(cnn_model))
    if (not config["train_new_cnn_model"] and
            os.path.isfile(config["cnn_model_path"])):
        print("Loading cnn model...")
        cnn_model.load_state_dict(load_torch_model(config["cnn_model_path"]))
        cnn_model.eval()
        return cnn_model

    print("Training cnn model...")
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")

    criterion = CrossEntropyLoss()
    optimizer_class = get_optimizer_class(config["cnn_optimizer"])
    optimizer = optimizer_class(cnn_model.parameters(), lr=config["cnn_lr"])
    lr_scheduler = get_lr_scheduler(optimizer,
                                    config["cnn_epoch_n"],
                                    verbose=True)

    # _, best_model = train_eval_loop(
    #     cnn_model,
    #     train_dataset,
    #     test_dataset,
    #     BCELoss(),
    #     lr=config["cnn_lr"],
    #     epoch_n=config["cnn_epoch_n"],
    #     batch_size=config["cnn_batch_size"],
    #     # device="cuda",
    #     device=config["device"],
    #     early_stopping_patience=config["cnn_early_stopping_patience"],
    #     max_batches_per_epoch_train=config["cnn_max_batches_per_epoch_train"],
    #     max_batches_per_epoch_val=len(test_dataset),
    #     optimizer_ctor=Adagrad,
    #     # lr_scheduler_ctor=lambda optim: lr_scheduler.
    #     # shuffle_train=False,
    #     # ReduceLROnPlateau(optim, patience=3),
    #     dataloader_workers_n=0,
    # )

    device = torch_device(config["device"])

    trainer = Trainer(
        model=cnn_model,
        epochs=config["cnn_epoch_n"],
        train_dataloader=train_dataset,
        train_steps=config["cnn_batch_size"],
        val_dataloader=val_dataset,
        val_steps=config["cnn_batch_size"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=None,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["cnn_model_dir"],
        model_name=config["cnn_model_name"],
    )

    trainer.train()

    trainer.save_model()
    trainer.save_loss()

    save_config(config, config["model_dir"])

    return trainer.model


def train(config, rng_numpy):
    data_plot, data_quote = read_data(config)

    # get text data
    data, labels = generate_data_labels(data_plot, data_quote)
    data, labels = shuffle_data_labels(data, labels, rng_numpy)

    train_size = int(len(data) * config["train_size"])

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
                                     max_doc_freq=config["vocab_max_doc_freq"],
                                     min_count=config["vocab_min_count"],
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
    tagger = load_spacy(
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

    model = train_cnn(train_dataset, test_dataset, config)
    save_vocab(vocabulary, config["model_dir"])
    accuracy = check_accuracy(model, train_dataset, test_dataset)

    with open("output.txt", "a") as file:
        file.write(str({**accuracy, **config, "time": datetime.now()}) + "\n")


if __name__ == "__main__":
    config = setup_config()
    init_random_seed()
    generator = RandomGenerator()
    train(config, generator.rng_numpy)
