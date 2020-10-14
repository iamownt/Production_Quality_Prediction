import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from utils import *
import sys
sys.path.extend(['D:\\Github\\pytorch-fm', 'D:/Github/pytorch-fm'])
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    """Create a LSTMEncoder for the MaskModel。
    Args:
        input_dim: TODO
        hidden_dim: TODO
    """
    def __init__(self, input_dim, hidden_dim, step, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.step = step
        self.cell = nn.LSTMCell(input_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c

    def forward(self, time_series):
        """
        :param time_series: the expected time_series shape is (batch_size, time_step, feature_size)
        :return: h: last cell's hidden dim
        """
        batch_size = time_series.size(0)
        h, c = self.init_hidden_state(batch_size)

        for i in range(self.step):
            h, c = self.cell(time_series[:, i, :], (h, c))

        return c


class MaskAndConcat(nn.Module):

    def __init__(self, mask_rate, tr_in, tr_out):
        super(MaskAndConcat, self).__init__()
        self.mask_rate = mask_rate
        self.fc = nn.Linear(tr_in, tr_out)

    def forward(self, con_in, h_in):
        # con_in(B, step*D)
        h_tr = self.fc(h_in)  # (B, h_tr)
        mask = torch.rand(*con_in.size()) > self.mask_rate

        return torch.concat([con_in*mask, h_tr], dim=1), mask


class DeepComponent(nn.Module):

    """
    注意模型的输入维度是shape(con_in) + shape(h_tr)， 不能小于中间层，
    """
    def __init__(self, input_dim, hidden_dims, dropout):
        super(DeepComponent, self).__init__()
        layers = list()
        for hidden_dim in (hidden_dims + hidden_dims[::-1]):
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.LeakyReLU(0.8))
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = hidden_dim
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class RecLoss(nn.Module):

    def __init__(self, bernoulli=True):
        super(RecLoss, self).__init__()
        self.bernoulli = bernoulli

    def forward(self, preds, labels, mask):

        y1 = preds * mask
        y2 = labels * mask
        error = y1 - y2

        return torch.mean(torch.square(error/y2.std(dim=0)))











