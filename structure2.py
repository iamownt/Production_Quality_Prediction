import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    """Create a LSTMEncoder for the structure1。
    Args:
        input_dim: TODO
        hidden_dim: TODO
    """
    def __init__(self, input_dim, hidden_dim, step, drop_out=0.5):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.step = step
        self.dropout_rate = drop_out
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, 1) for i in range(self.step)])
        self.cell = nn.LSTMCell(input_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c

    def forward(self, time_series):
        """
        :param time_series: the expected time_series shape is (batch_size, time_step, feature_size)
        :return: h: last cell's hidden dim
        :return: out_1: (batch_size, 5)
        """
        batch_size = time_series.size(0)
        h, c = self.init_hidden_state(batch_size)
        out_1 = []

        for i in range(self.step):
            h, c = self.cell(time_series[:, i, :], (h, c))
            out_1.append(self.linears[i](c))  # append (B*1)

        out_1 = torch.cat(out_1, dim=1)
        return h, out_1