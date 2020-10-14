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


class DNNDecoder(nn.Module):
    """
    with the hidden component's shape 512
    """

    def __init__(self, D_in, H1, H2, hidden_size, dropout=0.5):
        super(DNNDecoder, self).__init__()

        self.D_in = D_in
        self.H1 = H1
        self.H2 = H2
        self.dropout_rate = dropout

        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(2*H1, H2)
        self.relu_tr = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.linear_tr = nn.Linear(hidden_size, self.H1)

    def forward(self, x, hidden_component):

        # 不能把linear_tr定义在这里，随着输入决定，因为如果这样子，每次调用forward方法就会重置一次nn.Linear!!!!
        hidden = self.relu_tr(self.linear_tr(hidden_component))
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        x = torch.cat([x, hidden], dim=1)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu2(x)
        # x = self.fc3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.dropout2(x)
        return x


class EncoderDecoderModel(nn.Module):

    """
    :Arg：
    x_left:expect size is (B, step, feature_size)
    x_right:expect size is (B, FEATURE)
    note FEATURE is composed of:
        1_hour_delay feature: feature_size
        2_hour_delay feature and label: feature_size + 1
    """

    def __init__(self, input_dim, hidden_dim, step, D_in, H1, H2, D_out=1, drop_out_1=0.2, drop_out_2=0.2):
        super(EncoderDecoderModel, self).__init__()
        self.D_out = D_out
        self.encoder = LSTMEncoder(input_dim, hidden_dim, step, drop_out=drop_out_1)
        self.decoder = DNNDecoder(D_in, H1, H2, dropout=drop_out_2)
        self.fc = nn.Linear(H2 + step, D_out)

    def forward(self, x_left, x_right):
        lstm_out, lstm_hidden = self.encoder(x_left)
        dnn_out = self.decoder(x_right, lstm_hidden)
        out = torch.cat([lstm_out, dnn_out], dim=1)
        x = self.fc(out, self.D_out)
        return x
