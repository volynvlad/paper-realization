from datetime import datetime
import traceback
import copy
from typing import List

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def read_data(config):
    with open(config["plot_path"], "rb") as read:
        data_plot = read.read().decode(errors="replace").split("\n")

    with open(config["quote_path"], "rb") as read:
        data_quote = read.read().decode(errors="replace").split("\n")

    return np.array(data_plot[:-1]), np.array(data_quote[:-1])


def generate_data_labels(*args):
    data = np.array([], dtype=np.int32)
    labels = np.array([], dtype=np.int32)
    for i, arg in enumerate(args):
        data = np.concatenate([data, arg])
        labels = np.concatenate(
            [labels, np.ones_like(arg, dtype=np.int32) * int(i)])
    return data, labels


def shuffle_data_labels(data, labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return data[indices], labels[indices]


def ensure_length(txt, out_len, pad_value):
    if len(txt) < out_len:
        txt = list(txt) + [pad_value] * (out_len - len(txt))
    else:
        txt = txt[:out_len]
    return np.array(txt)


class PaddedSequenceDataset(Dataset):
    def __init__(self,
                 texts,
                 targets,
                 embeddings=None,
                 out_len=30,
                 pad_value=0):
        self.texts = texts
        self.targets = targets
        self.embeddings = embeddings
        self.out_len = out_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.texts)

    @property
    def shape(self):
        return self.texts.shape

    def __getitem__(self, item):
        txt = self.texts[item]
        # txt = np.array([ensure_length(t, self.out_len, self.pad_value)
        #                 for t in txt])
        txt = ensure_length(txt, self.out_len, self.pad_value)
        if self.embeddings is None:
            # txt = torch.tensor(txt, dtype=torch.float)
            txt = np.array(txt)
        else:
            txt = self.embeddings.get_vectors(txt)
        target = torch.tensor(self.targets[item], dtype=torch.float)

        return txt, target


def make_diag_mask(size, radius):

    idxs = torch.arange(size)
    abs_idx_diff = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs()
    mask = ((abs_idx_diff <= radius) & (abs_idx_diff > 0)).float()
    return mask


class SkipGramNegativeSamplingTrainer(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_size,
                 sentence_len,
                 radius=5,
                 negative_samples_n=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.negative_samples_n = negative_samples_n

        self.center_emb = nn.Embedding(self.vocab_size,
                                       emb_size,
                                       padding_idx=0)
        self.center_emb.weight.data.uniform_(-1.0 / emb_size, 1.0 / emb_size)
        self.center_emb.weight.data[0] = 0

        self.context_emb = nn.Embedding(self.vocab_size,
                                        emb_size,
                                        padding_idx=0)
        self.context_emb.weight.data.uniform_(-1.0 / emb_size, 1.0 / emb_size)
        self.context_emb.weight.data[0] = 0

        self.positive_sim_mask = make_diag_mask(sentence_len, radius)

    def forward(self, sentences):
        """
		sentences - Batch x MaxSentLength - tokens identificators
		"""

        batch_size = sentences.shape[0]
        # Batch x MaxSentLength x EmbSize
        center_embeddings = self.center_emb(sentences)

        # Batch x EmbSizex MaxSentLength
        positive_context_embs = self.context_emb(sentences).permute(0, 2, 1)
        # Batch x MaxSentLength x MaxSentLength
        positive_sims = torch.bmm(center_embeddings, positive_context_embs)
        positive_probs = torch.sigmoid(positive_sims)

        positive_mask = self.positive_sim_mask.to(positive_sims.device)
        positive_loss = F.binary_cross_entropy(
            positive_probs * positive_mask,
            positive_mask.expand_as(positive_probs))

        # Batch x NegSamplesN
        negative_words = torch.randint(1,
                                       self.vocab_size,
                                       size=(batch_size,
                                             self.negative_samples_n),
                                       device=sentences.device)

        # Batch x EmbSize x NegSamplesN
        negative_context_embs = self.context_emb(negative_words).permute(
            0, 2, 1)

        # Batch x MaxSentLength x NegSamplesN
        negative_sims = torch.bmm(center_embeddings, negative_context_embs)
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_sims, negative_sims.new_zeros(negative_sims.shape))

        return positive_loss + negative_loss


def no_loss(pred, target):
    return pred


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return np.array([copy_data_to_device(elem, device) for elem in data], dtype=object)
    raise ValueError(f"Unreachable data type: {type(data)}")


def train_eval_loop(model,
                    train_dataset,
                    val_dataset,
                    criterion,
                    lr=1e-4,
                    epoch_n=10,
                    batch_size=32,
                    device=None,
                    early_stopping_patience=10,
                    l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0,
                    is_embedding=False):
    """
    A cycle for training models.
        After each epoch,
        the evaluation of the quality of the model is delayed by the sample.
    :param model: torch.nn.Module - training model
    :param train_dataset: torch.utils.data.Dataset - data for training
    :param val_dataset: torch.utils.data.Dataset - data for validation
    :param criterion: loss function
    :param lr: learning rate
    :param epoch_n: max number of epochs
    :param batch_size: number of samples,
        that are training by model on each iteration
    :param device: cuda/cpu - device on which we will train model
    :param early_stopping_patience: the largest number of epochs during which
        lack of model improvement to keep learning going.
    :param l2_reg_alpha: coef L2-regularization
    :param max_batches_per_epoch_train:
        max number of iteration on each epoch on train
    :param max_batches_per_epoch_val:
        max number of iteration on each epoch on valid
    :param data_loader_ctor:
        function of the object that convert dataset into batches
    (default: torch.utils.data.DataLoader)
    :return: tuple of two elements:
        - mean loss on validation on best epoch
        - best model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float("inf")
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.now()
            print(f"Epoch {epoch_i}")

            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)
                if not is_embedding and (isinstance(pred, torch.Tensor)
                                         and isinstance(batch_y, torch.Tensor)
                                         and len(pred.shape) != len(batch_y.shape)):
                    pred = torch.reshape(pred, batch_y.shape)

                try:
                    loss = criterion(pred, batch_y)
                except:
                    print(f"{pred=}, {batch_y=}")
                    print(f"{len(pred)=}, {len(batch_y)=}")
                    raise ValueError

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            time_delta = (datetime.now() - epoch_start).total_seconds()
            print(f"Epoch: {train_batches_n} "
                  f"iterations, {time_delta:0.2f} sec")
            print("Mean value of loss "
                  f"function on training: {mean_train_loss}")

            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    if not is_embedding and (isinstance(pred, torch.Tensor)
                                             and isinstance(batch_y,
                                                            torch.Tensor)
                                             and len(pred.shape) != len(batch_y.shape)):
                        batch_y = torch.reshape(batch_y, pred.shape)

                    try:
                        loss = criterion(pred, batch_y)
                    except:
                        print(f"{pred=}, {batch_y=}")
                        print(f"{len(pred)=}, {len(batch_y)=}")
                        raise ValueError

                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print("Mean value of loss function "
                  f"on validation: {mean_val_loss}")

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print("New best model!")
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print("Models does not evolve in "
                      f"{early_stopping_patience} epochs, cancel training")
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print("Cancel training")
            break
        except Exception as ex:
            print("Error while training: {}\n{}".format(
                ex, traceback.format_exc()))
            break

    return best_val_loss, best_model


class Embeddings:
    def __init__(self, embeddings, word2id, tagger=None):
        self.embeddings = embeddings
        self.embeddings /= (
            np.linalg.norm(embeddings,
                           ord=2,
                           axis=-1,
                           keepdims=True) +
            1e-4)
        self.word2id = word2id
        self.id2word = {i: w for w, i in word2id.items()}
        self.tagger = tagger

    def __len__(self):
        return len(self.embeddings)

    @property
    def shape(self):
        return self.embeddings.shape

    def most_similar(self, word, topk=10):
        print(f"Most similar to {word} is :")
        return self.most_similar_by_vector(self.get_vector(word), topk=topk)

    def analogy(self, positive, negative, topk=10):
        assert len(positive) == 2
        assert len(negative) == 1
        a1_v = self.get_vector(negative[0])
        b1_v = self.get_vector(positive[0])
        a2_v = self.get_vector(positive[1])
        query = b1_v - a1_v + a2_v
        return self.most_similar_by_vector(query, topk=topk)

    def most_similar_by_vector(self, query_vector, topk=10):
        similarities = (self.embeddings * query_vector).sum(-1)
        best_indices = np.argpartition(-similarities, topk, axis=0)[:topk]
        result = [(self.id2word[i], similarities[i]) for i in best_indices]
        result.sort(key=lambda pair: -pair[1])
        return result

    def get_vector(self, word):
        assert (isinstance(word, str)
                or isinstance(word, torch.Tensor)
                or isinstance(word, np.ndarray))
        if isinstance(word, torch.Tensor):
            return np.array(self.embeddings[word.tolist()])
        else:
            if self.tagger:
                tokens = self.tagger(word)
                print(
                    f"Get vector to the word = {word}"
                    f"and pos tag = {tokens[0].pos_}"
                )
                word = word + "_" + tokens[0].pos_
            if word not in self.word2id:
                raise ValueError(f'Unknown word "{word}"')
            return np.array(self.embeddings[self.word2id[word]])

    def get_vectors(self, words):
        assert (isinstance(words, List)
                or isinstance(words, torch.Tensor)
                or isinstance(words, np.ndarray))
        if isinstance(words, np.ndarray):
            word_ids = np.stack([w for w in words])
        elif isinstance(words, torch.Tensor):
            word_ids = np.stack([w.tolist() for w in words])
        else:
            if self.tagger:
                words = [
                    "{}_{}".format(token.text, token.pos_)
                    for token in self.tagger(" ".join(words))
                ]
            word_ids = np.stack([self.word2id[w] for w in words])
        vectors = np.array([np.array(self.embeddings[w])
                            for w in word_ids])
        return vectors
