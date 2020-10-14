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
sys.path.extend(['/Users/wt/Documents/GitHub/pytorch-fm'])
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

    def __init__(self, mask_c, mask_h, tr_in, tr_out):
        super(MaskAndConcat, self).__init__()
        self.mask_c = mask_c
        self.mask_h = mask_h
        self.fc = nn.Linear(tr_in, tr_out)

    def forward(self, con_in, h_in):
        # con_in(B, step*D)
        h_tr = self.fc(h_in)  # (B, h_tr)
        h_tr = (h_tr - torch.mean(h_tr, dim=0))/torch.std(h_tr, dim=0)
        mask1 = torch.rand(*con_in.size()) >= self.mask_c
        mask2 = torch.rand(*h_tr.size()) >= self.mask_h

        ori = torch.cat([con_in*mask1, h_tr*mask2], dim=1)
        masked = torch.cat([con_in, h_tr], dim=1)
        return ori, masked, mask1, mask2


class DeepComponent(nn.Module):

    """
    注意模型的输入维度是shape(con_in) + shape(h_tr)， 不能小于中间层，
    """
    def __init__(self, input_dim, hidden_dims, dropout):
        super(DeepComponent, self).__init__()
        layers = list()
        input_ori = input_dim
        for hidden_dim in (hidden_dims + hidden_dims[::-1]):
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.LeakyReLU(0.8))
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, input_ori))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class RecLoss(nn.Module):

    def __init__(self, df_std, h_dim, bernoulli=True):
        super(RecLoss, self).__init__()
        self.bernoulli = bernoulli
        self.std_1 = torch.from_numpy(df_std.values).float()
        self.std_1 = torch.cat([self.std_1]*5)
        self.std_2 = torch.ones(h_dim)

    def forward(self, flag, preds, labels, mask):

        y1 = preds * (1 - mask)
        y2 = labels * (1 - mask)
        error = y1 - y2

        return torch.mean((error/self.std_1)**2) if flag else torch.mean((error/self.std_2)**2)


class PretrainDataset(Dataset):
    """
    A Pytorch Dataset class to be used in PyTorch DataLoader to create batches
    """
    def __init__(self, data_des, step):
        self.step = step
        # self.df = pd.read_csv(data_des).iloc[:128*180,:]
        self.df = pd.read_csv(data_des)
        col_leave = ['% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
                     'Flotation Column 07 Air Flow', 'Flotation Column 03 Level', 'Flotation Column 07 Level',
                     '% Silica Concentrate', '2_hour_delay', '3_hour_delay']
        col_list = [col for col in self.df.columns if col in col_leave]
        self.df = self.df[col_list]
        self.dataset_size = len(self.df) - self.step + 1

    def __getitem__(self, i):
        """
        :param i:
        :return:  (time_step. feature_size)
        """
        x_ori = torch.from_numpy(self.df.iloc[i:i + self.step, :-1].values).float().to(device)
        return x_ori

    def __len__(self):
        return self.dataset_size


class EncoderDecoderModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, step, mask_c, mask_h, hidden_tr, hidden_list, dropout):
        super(EncoderDecoderModel, self).__init__()
        self.lstm = LSTMEncoder(input_dim, hidden_dim, step)
        self.mask_concat = MaskAndConcat(mask_c, mask_h, hidden_dim, hidden_tr)
        self.deep = DeepComponent(step*input_dim + hidden_tr, hidden_list, dropout)
        assert(step*input_dim + hidden_tr > hidden_list[-1])

    def forward(self, time_series):

        batch_size = time_series.size()[0]
        c_last = self.lstm(time_series)
        time_re = time_series.reshape(batch_size, -1)
        ori, con_inp, mask1, mask2 = self.mask_concat(time_re, c_last)
        deep_out = self.deep(con_inp)

        return deep_out, ori, mask1, mask2


input_dim = 11
hidden_dim = 512
step = 5
batch_size = 128
hidden_tr = 64
mask_c = 0.15
mask_h = 0.1
mlp_list = (512, 256, 64)
dropout = 0
epochs = 500
grad_clip = 5.
print_freq = 10
epochs_since_improvement = 0
output_folder = "/Users/wt/Downloads/MiningProcessEngineering"
train_left_des = "/Users/wt/Downloads/MiningProcessEngineering/train_left_now.csv"

df = pd.read_csv(train_left_des)
col_leave = ['% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
             'Flotation Column 07 Air Flow', 'Flotation Column 03 Level', 'Flotation Column 07 Level',
             '2_hour_delay', '3_hour_delay']
col_list = [col for col in df.columns if col in col_leave]
df = df[col_list]

ende_model = EncoderDecoderModel(input_dim, hidden_dim, step, mask_c, mask_h, hidden_tr, mlp_list, dropout)
criterion = RecLoss(df.std(), hidden_tr).to(device)
optimizer = Adam(ende_model.parameters(), lr=3e-4)
train_loader = DataLoader(PretrainDataset(train_left_des, step), batch_size=batch_size, shuffle=True, pin_memory=True)
best_loss = 999
for epoch in range(epochs):
    if epochs_since_improvement == 10:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
        adjust_learning_rate(optimizer, 0.8)
    ende_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    for i, time_series in enumerate(train_loader):
        data_time.update(time.time() - start)
        time_series.to(device)
        re_con, ori_con, mask1, mask2 = ende_model(time_series)
        loss1 = criterion(1, re_con[:, :-hidden_tr], ori_con[:, :-hidden_tr], mask_c)
        loss2 = criterion(0, re_con[:, -hidden_tr:], ori_con[:, -hidden_tr:], mask_h)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()

        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()
        # print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses))



nn.MSELoss()