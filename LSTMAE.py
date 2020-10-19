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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMEncoder(nn.Module):
    """Create a Two Layer LSTMEncoder for the MaskModel。
    Args:
        input_dim: TODO
        hidden_dim: TODO
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, step):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.step = step
        self.cell1 = nn.LSTMCell(input_dim, hidden_dim, bias=True)
        self.cell2 = nn.LSTMCell(hidden_dim, embedding_dim, bias=True)

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
        assert(embedding_dim < input_dim * step)  # or the task will be so easy
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

        return torch.stack(recon_lis).permute(1, 0, 2)

class LSTMEncoderDecoder(nn.Module):
    """
    Use simple two layer LSTM.
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, step):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, embedding_dim, step)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, embedding_dim, step)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

def load_checkpoint(model, checkpoint, optimizer, loadOptimizer):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint, map_location='cpu')
        pretrained_dict = modelCheckpoint['state_dict']
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
        # 如果不需要更新优化器那么设置为false
        if loadOptimizer:
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print('loaded! optimizer')
        else:
            print('not loaded optimizer')
    else:
        print('No checkpoint is included')
    return model, optimizer


input_dim = 11
hidden_dim = 256
embedding_dim = 24 # very important parameters to use.
step = 5
batch_size = 512
epochs = 100
grad_clip = 5.
print_freq = 50
epochs_since_improvement = 0
output_folder = "/Users/wt/Downloads/MiningProcessEngineering"
train_left_des = "/Users/wt/Downloads/MiningProcessEngineering/train_left_now.csv"
val_left_des = "/Users/wt/Downloads/MiningProcessEngineering/test_left_now.csv"
# output_folder = r"D:\Datasets\Mining_output"
# checkpoint_path = r"D:\Datasets\Mining_output\checkpoint"
# train_left_des = r"D:\Datasets\MiningProcessEngineering\train_left_now.csv"
# val_left_des = r"D:\Datasets\MiningProcessEngineering\test_left_now.csv"


lstm_ed = LSTMEncoderDecoder(input_dim, hidden_dim, embedding_dim, step).to(device)
optimizer = Adam(lstm_ed.parameters(), lr=3e-4)
criterion = nn.MSELoss().to(device)
train_loader = DataLoader(PretrainDataset(train_left_des, step), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(PretrainDataset(val_left_des, step), batch_size=batch_size, shuffle=True)
# shuffle=False
best_loss = 999
for epoch in range(epochs):
    if epochs_since_improvement == 10:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
        adjust_learning_rate(optimizer, 0.8)
    lstm_ed.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    for i, time_series in enumerate(train_loader):
        data_time.update(time.time() - start)
        time_series.to(device)
        re_con = lstm_ed(time_series)
        loss = criterion(re_con, time_series)
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
    lstm_ed.eval()
    batch_time = AverageMeter()
    val_losses = AverageMeter()
    start = time.time()
    with torch.no_grad():

        for i, time_series in enumerate(val_loader):
            time_series.to(device)
            re_con = lstm_ed(time_series)
            loss = criterion(re_con, time_series)
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
            torch.save({'epoch': epoch+1, 'state_dict': lstm_ed.state_dict(), 'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()}, os.path.join(checkpoint_path,
                                                                           str("%d_%.4f.pth.tar" % (epoch, best_loss))))
        else:
            epochs_since_improvement += 1
        print('Validation: [{0}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i,
                                                              batch_time=batch_time,
                                                              loss=val_losses))