import os
import argparse
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import embedding, nn
import spacy
import scipy.sparse

from utils import tokenize_corpus, build_vocabulary, texts2token_ids, timing
from data import read_data, generate_data_labels, shuffle_data_labels
from data import PaddedSequenceDataset, \
  SkipGramNegativeSamplingTrainer,\
  Embeddings,\
  train_eval_loop,\
  no_loss,\
  copy_data_to_device

np.random.seed(373)


class CNNSentanceClassifier(nn.Module):
    def __init__(self,
                 h: int,
                 features_num: int,
                 sentence_length: int,
                 device=None):
        """_summary_

        Args:
            h (int): window size
            features_num (int): embedding size
            n_classes (int): number of classes
            device (_type_, optional): cpu or gpu. Defaults to None.
        """
        super(CNNSentanceClassifier, self).__init__()
        self.h = h
        self.sentence_length = sentence_length
        self.convolution = nn.Conv1d(sentence_length,
                                     sentence_length - h + 1,
                                     1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(features_num, stride=1)
        self.linear = nn.Linear(in_features=sentence_length - h + 1,
                                out_features=1,
                                bias=True)
        self.dropout = nn.Dropout(0.5)

        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)

    def __str__(self):
        res = f"""\n
{'=' * 100}\n
Network: {self._get_name()}\n
{'=' * 100}\n
Convolution: {self.convolution}\n
{'-' * 100}\n
MaxPool: {self.max_pool}\n
{'-' * 100}\n
Dropout: {self.dropout}\n
{'=' * 100}\n
Linear: {self.linear}\n
{'-' * 100}\n
        """
        return res

    # @timing
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = copy_data_to_device(x, self.device)
        x = self.relu(self.convolution(x))
        x = self.max_pool(x)
        x = torch.reshape(x, (-1, self.sentence_length - self.h + 1))
        x = self.dropout(x)
        x = self.linear(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to yaml config')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    data_plot, data_quote = read_data(config)

    # get text data
    data, labels = generate_data_labels(data_plot, data_quote)
    data, labels = shuffle_data_labels(data, labels)

    TRAIN_VAL_SPLIT = int(len(data) * 0.7)

    # split text data
    train_data = data[:TRAIN_VAL_SPLIT]
    test_data = data[TRAIN_VAL_SPLIT:]
    train_labels = labels[:TRAIN_VAL_SPLIT]
    test_labels = labels[TRAIN_VAL_SPLIT:]

    # tokenize text data
    train_tokenized = tokenize_corpus(train_data)
    test_tokenized = tokenize_corpus(test_data)

    # build vocabulary from train text data
    vocabulary, word_doc_freq = build_vocabulary(train_tokenized,
                                                 max_doc_freq=0.85,
                                                 min_count=3,
                                                 pad_word="<PAD>")
    print(f"vocabulary size = {len(vocabulary)}")

    # make ids from tokens
    train_token_ids = texts2token_ids(train_tokenized, vocabulary)
    test_token_ids = texts2token_ids(test_tokenized, vocabulary)

    # sent_length = [len(s) for s in train_token_ids]
    # plt.hist(sent_length, bins=20)
    # plt.title("histogram")
    # plt.show()

    # print(pd.DataFrame(sent_length).describe())
    # print(np.percentile(sent_length, 80))
    # print(np.percentile(sent_length, 90))
    # print(np.percentile(sent_length, 95))
    # print(np.percentile(sent_length, 97))
    # print(np.percentile(sent_length, 99))

    # add paddings to data
    train_dataset = PaddedSequenceDataset(train_token_ids,
                                          train_labels,
                                          out_len=config["max_sentence_len"])

    test_dataset = PaddedSequenceDataset(test_token_ids,
                                         test_labels,
                                         out_len=config["max_sentence_len"])

    @timing
    def train_word2vec(config):

        trainer = SkipGramNegativeSamplingTrainer(len(vocabulary),
                                                  config["features_num"],
                                                  config["max_sentence_len"],
                                                  radius=3,
                                                  negative_samples_n=32)
        if not config['train_new_word2vec_model'] and os.path.isfile(config["word2vec_model_path"]):

            trainer.load_state_dict(torch.load(config['word2vec_model_path']))
            return trainer
        else:

            print(f"train dataset size: {len(train_dataset)}")
            print(f"test dataset size: {len(test_dataset)}")

            best_val_loss, best_model = train_eval_loop(
                trainer,
                train_dataset,
                test_dataset,
                no_loss,
                lr=3e-4,
                epoch_n=200,
                batch_size=32,
                device=config["device"],
                early_stopping_patience=20,
                max_batches_per_epoch_train=3000,
                max_batches_per_epoch_val=len(test_dataset),
                lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.
                ReduceLROnPlateau(optim, patience=5))

            torch.save(best_model.state_dict(), config['word2vec_model_path'])

            return best_model

    trainer = train_word2vec(config)
    # tagger = spacy.load(
    #     "en_core_web_sm",
    # 	disable=['tok2vec', 'parser', 'senter', 'lemmatizer', 'ner'])
    embeddings = Embeddings(trainer.center_emb.weight.detach().cpu().numpy(),
                            vocabulary,
                            tagger=None)

    # assert (np.array(embeddings.get_vectors(train_dataset[0][0])) == np.array([embeddings.get_vector(x) for x in train_dataset[0][0]])).all()

    train_dataset = PaddedSequenceDataset(train_token_ids,
                                          train_labels,
                                          embeddings=embeddings,
                                          out_len=config["max_sentence_len"])
    test_dataset = PaddedSequenceDataset(test_token_ids,
                                         test_labels,
                                         embeddings=embeddings,
                                         out_len=config["max_sentence_len"])

    # print(embeddings.most_similar("actor"))
    # print(embeddings.most_similar("film"))
    # print(embeddings.most_similar("good"))
    # print(embeddings.most_similar("terrible"))

    # exit()

    @timing
    def train_cnn(config):

        cnn_model = CNNSentanceClassifier(h=config["window_size"],
                                          features_num=config["features_num"],
                                          sentence_length=config["max_sentence_len"],
                                          device="cpu")
        if (not config['train_new_cnn_model'] and
            os.path.isfile(config["cnn_model_path"])):

            cnn_model.load_state_dict(torch.load(config['cnn_model_path']))
            return cnn_model
        else:

            print(f"train dataset size: {len(train_dataset)}")
            print(f"test dataset size: {len(test_dataset)}")

            best_val_loss, best_model = train_eval_loop(
                cnn_model,
                train_dataset,
                test_dataset,
                torch.nn.CrossEntropyLoss(),
                lr=3e-4,
                epoch_n=200,
                batch_size=32,
                # device="cuda",
                device=config["cpu"],
                early_stopping_patience=20,
                max_batches_per_epoch_train=3000,
                max_batches_per_epoch_val=len(test_dataset),
                lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.
                ReduceLROnPlateau(optim, patience=5))

            torch.save(cnn_model.state_dict(), config['cnn_model_path'])

            print(f"{best_val_loss=}")

            return best_model

    train_cnn(config)


if __name__ == "__main__":
    main()
