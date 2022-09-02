"""

class CNNSentenceClassifier

"""
from typing import Union

import numpy as np
from torch import nn, from_numpy, device as torch_device, Tensor, reshape
from torch.cuda import is_available

from utils.data import copy_data_to_device


#  TODO make for CUDA device
class CNNSentenceClassifier(nn.Module):
    """

    CNNSentenceClassifier class
    override nn.Module

    """
    def __init__(self,
                 window_size: int,
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
        super().__init__()
        self.window_size = window_size
        self.sentence_length = sentence_length
        self.conv_block = nn.Sequential(
            nn.Conv1d(sentence_length,
                      sentence_length - window_size + 1,
                      1),
            nn.ReLU(),
            nn.MaxPool1d(features_num, stride=2),
            nn.Dropout(0.5),
        )
        self.linear_block = nn.Sequential(
            nn.Linear(in_features=sentence_length - window_size + 1,
                      out_features=1,
                      bias=True),
            nn.Sigmoid(),
        )
        self.device = device
        if self.device is None:
            self.device = "cuda" if is_available() else "cpu"
        self.device = torch_device(self.device)

    def __str__(self):
        res = f"""\n
{"=" * 100}\n
Network: {self._get_name()}\n
{"=" * 100}\n
Convolution block: {self.conv_block}\n
{"-" * 100}\n
Linear block: {self.linear_block}\n
{"-" * 100}\n
        """
        return res

    # @timing
    def forward(self, tensor: Union[np.ndarray, Tensor]):
        """_summary_

        Args:
            tensor (_type_): _description_

        Returns:
            _type_: _description_
        """
        tensor = copy_data_to_device(tensor, self.device)
        tensor = self.conv_block(tensor)
        tensor = reshape(tensor,
                         (-1,
                          self.sentence_length - self.window_size + 1))
        tensor = self.linear_block(tensor)
        return tensor
