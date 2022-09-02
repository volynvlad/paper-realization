from torch import nn, bmm, sigmoid, randint
from torch.nn import functional as F

from utils.data import make_diag_mask


class SkipGramNegativeSamplingTrainer(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_size,
                 sentence_len,
                 radius=5,
                 negative_samples_n=5,
                 ):
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

        # Batch x EmbSize x MaxSentLength
        positive_context_embs = self.context_emb(sentences).permute(0, 2, 1)
        # Batch x MaxSentLength x MaxSentLength
        positive_sims = bmm(center_embeddings, positive_context_embs)
        positive_probs = sigmoid(positive_sims)

        positive_mask = self.positive_sim_mask.to(positive_sims.device)
        positive_loss = F.binary_cross_entropy(
            positive_probs * positive_mask,
            positive_mask.expand_as(positive_probs))

        # Batch x NegSamplesN
        negative_words = randint(1,
                                 self.vocab_size,
                                 size=(batch_size,
                                       self.negative_samples_n),
                                 device=sentences.device)
        # Batch x EmbSize x NegSamplesN
        negative_context_embs = self.context_emb(negative_words).permute(
            0, 2, 1)

        # Batch x MaxSentLength x NegSamplesN
        negative_sims = bmm(center_embeddings, negative_context_embs)
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_sims, negative_sims.new_zeros(negative_sims.shape))

        return positive_loss + negative_loss


def no_loss(pred, target):
    return pred


class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self,
                 vocab_size: int,
                 emb_size):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            max_norm=1,
        )
        self.linear = nn.Linear(
            in_features=emb_size,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self,
                 vocab_size: int,
                 emb_size):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            max_norm=1,
        )
        self.linear = nn.Linear(
            in_features=emb_size,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
