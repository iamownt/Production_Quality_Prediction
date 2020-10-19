import time
import os
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    """Create a Two Layer LSTMEncoder for the MaskModel。
    Args:
        input_dim: TODO
        hidden_dim: TODO
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, step, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.step = step
        self.cell1 = nn.LSTMCell(input_dim, hidden_dim, bias=True)
        self.cell2 = nn.LSTMCell(hidden_dim, embedding_dim, bias=True)
        # self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, batch_size, hidden_dim):
        h = torch.zeros(batch_size, hidden_dim).to(device)
        c = torch.zeros(batch_size, hidden_dim).to(device)
        return h, c

    def forward(self, time_series):
        """
        :param time_series: the expected time_series shape is (batch_size, time_step, feature_size)
        :return: h: last cell's hidden dim
        """
        batch_size = time_series.size(0)
        h0, c0 = self.init_hidden_state(batch_size, self.hidden_dim)
        h1, c1 = self.init_hidden_state(batch_size, self.embedding_dim)

        for i in range(self.step):
            h0, c0 = self.cell1(time_series[:, i, :], (h0, c0))
            h1, c1 = self.cell2(h0, (h1, c1))

        return h1


class LSTMDecoder(nn.Module):
    """Create a Two Layer LSTMDecoder"""

    def __init__(self, input_dim, hidden_dim, embedding_dim, step):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.step = step
        self.cell1 = nn.LSTMCell(embedding_dim, hidden_dim, bias=True)
        self.cell2 = nn.LSTMCell(hidden_dim, input_dim, bias=True)

    def init_hidden_state(self, batch_size, hidden_dim):
        h = torch.zeros(batch_size, hidden_dim).to(device)
        c = torch.zeros(batch_size, hidden_dim).to(device)
        return h, c

    def forward(self, emb_inp):
        # 假定emb_inp是一个batch_size * embedding_size的矩阵，那么将它扩充
        recon_lis = []
        emb_inp = emb_inp.unsqueeze(1).repeat(1, self.step, 1)
        batch_size = emb_inp.size(0)
        h0, c0 = self.init_hidden_state(batch_size, self.hidden_dim)
        h1, c1 = self.init_hidden_state(batch_size, self.input_dim)

        for i in range(self.step):
            h0, c0 = self.cell1(emb_inp[:, i, :], (h0, c0))
            h1, c1 = self.cell2(h0, (h1, c1))
            recon_lis.append(h1)

        return recon_lis


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


input_dim = 11
hidden_dim = 512
step = 5
batch_size = 512
hidden_tr = 64
mask_c = 0.5
mask_h = 0.5
mlp_list = (512, 256, 64)
dropout = 0
epochs = 500
grad_clip = 5.
print_freq = 50
epochs_since_improvement = 0
val_samples = 99
# output_folder = "/Users/wt/Downloads/MiningProcessEngineering"
# train_left_des = "/Users/wt/Downloads/MiningProcessEngineering/train_left_now.csv"
output_folder = r"D:\Datasets\Mining_output"
checkpoint_path = r"D:\Datasets\Mining_output\checkpoint"
train_left_des = r"D:\Datasets\MiningProcessEngineering\train_left_now.csv"
val_left_des = r"D:\Datasets\MiningProcessEngineering\test_left_now.csv"
df = pd.read_csv(train_left_des)
col_leave = ['% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
             'Flotation Column 07 Air Flow', 'Flotation Column 03 Level', 'Flotation Column 07 Level',
             '2_hour_delay', '3_hour_delay']
col_list = [col for col in df.columns if col in col_leave]
df = df[col_list]

ende_model = EncoderDecoderModel(input_dim, hidden_dim, step, mask_c, mask_h, hidden_tr, mlp_list, dropout).to(device)
criterion = RecLoss(df.std(), hidden_tr, step).to(device)
optimizer = Adam(ende_model.parameters(), lr=3e-4)
train_loader = DataLoader(PretrainDataset(train_left_des, step), batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(PretrainDataset(val_left_des, step), batch_size=batch_size, shuffle=True, pin_memory=True)
# shuffle=False
best_loss = 999
for epoch in range(epochs):
    if epochs_since_improvement == 20:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
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
        loss_tr1 = criterion(1, re_con[:, :-hidden_tr], ori_con[:, :-hidden_tr], mask1)
        loss_tr2 = criterion(0, re_con[:, -hidden_tr:], ori_con[:, -hidden_tr:], mask2)
        loss = loss_tr1 + loss_tr2
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()

        losses.update(loss.item())
        batch_time.update(time.time() - start)
        # if i > print_freq:
        #     break

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
    # eval the model
    ende_model.eval()
    batch_time = AverageMeter()
    val_losses = AverageMeter()
    start = time.time()
    with torch.no_grad():

        for i, time_series in enumerate(val_loader):
            time_series.to(device)
            re_con, ori_con, mask1, mask2 = ende_model(time_series)
            loss_te1 = criterion(1, re_con[:, :-hidden_tr], ori_con[:, :-hidden_tr], mask1)
            loss_te2 = criterion(0, re_con[:, -hidden_tr:], ori_con[:, -hidden_tr:], mask2)
            loss = loss_te1 + loss_te2

            val_losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # if i >= val_samples:
            #     break
        val_mse = val_losses.avg
        # if val_mse < best_loss:
        if True:
            best_loss = val_mse
            epochs_since_improvement = 0
            torch.save({'epoch': epoch+1, 'state_dict': ende_model.state_dict(), 'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()}, os.path.join(checkpoint_path,
                                                                           str("%.4f.pth.tar" % best_loss)))
        else:
            epochs_since_improvement += 1
        print('Validation: [{0}/{1}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, val_samples,
                                                              batch_time=batch_time,
                                                              loss=val_losses))