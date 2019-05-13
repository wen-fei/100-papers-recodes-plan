from abc import ABC

import torch
from torch.autograd import Variable
from torch.nn import Module


class DecoderBase(ABC, Module):
    """

    :param ABC:
    :param Module:
    :return:
    """

    def forward(self, encoder_outputs, inputs, final_encoder_hidden,
                targets=None, teacher_forcing=1.0):
        raise NotImplementedError


def to_one_hot(y, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims
    and convert it to 1-hot representation with n+1 dims.
    :param y:
    :param n_dims:
    :return:
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
