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
# import sys
# # sys.path.extend(['/Users/wt/Documents/GitHub/pytorch-fm'])
# sys.path.extend(['D:\\Github\\pytorch-fm', 'D:/Github/pytorch-fm'])
# from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--mask_rate", type=float, default=0.15, help="mask_rate")
args = parser.parse_args()


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


class MaskAndConcat(nn.Module):
    """
    注意torch.rand放cuda里面。
    mask1.to(device)，实际上无效，得设置mask1=mask1.to(device)， 但是对于模型来说，只用model.eval()就相当于可以了，不用model=model.eval()
    """
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
        mask1 = mask1.to(device)
        mask2 = mask2.to(device)

        masked = torch.cat([con_in*mask1, h_tr*mask2], dim=1)
        ori = torch.cat([con_in, h_tr], dim=1)
        return ori, masked, mask1, mask2


class DeepComponent(nn.Module):

    """
    注意模型的输入维度是shape(con_in) + shape(h_tr)， 不能小于中间层，
    """
    def __init__(self, input_dim, hidden_dims, dropout):
        super(DeepComponent, self).__init__()
        layers = list()
        input_ori = input_dim

        for hidden_dim in (hidden_dims + hidden_dims[::-1][1:]):
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

    def __init__(self, df_std, h_dim, step, bernoulli=True):
        super(RecLoss, self).__init__()
        self.bernoulli = bernoulli
        self.std_1 = torch.from_numpy(df_std.values).float()
        self.std_1 = torch.cat([self.std_1]*step).to(device)
        self.std_2 = torch.ones(h_dim).to(device)

    def forward(self, flag, preds, labels, mask):

        y1 = preds * ~mask
        y2 = labels * ~mask
        count = torch.sum(~mask)
        error = y1 - y2
        return torch.sum((error/self.std_1)**2)/count if flag else torch.sum((error/self.std_2)**2)/count


class RecLoss_v2(nn.Module):

    def __init__(self, df_std, h_dim, step, alpha, belta, bernoulli=True):
        super(RecLoss_v2, self).__init__()
        self.alpha = alpha
        self.belta = belta
        self.bernoulli = bernoulli
        self.std_1 = torch.from_numpy(df_std.values).float()
        self.std_1 = torch.cat([self.std_1]*step).to(device)
        self.std_2 = torch.ones(h_dim).to(device)

    def forward(self, flag, preds, labels, mask):

        y1 = preds * ~mask
        y2 = labels * ~mask
        y3 = preds * mask
        y4 = labels * mask
        mlm_count = torch.sum(~mask)
        ae_count = torch.sum(mask)
        mlm_error = y1 - y2
        ae_error = y3 - y4
        mlm_loss = torch.sum((mlm_error/self.std_1)**2)/mlm_count if flag else torch.sum((mlm_error/self.std_2)**2)/mlm_count
        ae_loss = torch.sum((ae_error/self.std_1)**2)/ae_count if flag else torch.sum((ae_error/self.std_2)**2)/ae_count
        total_loss = self.alpha * mlm_loss + self.belta * ae_loss

        return total_loss


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

    def __init__(self, input_dim, hidden_dim, embedding_dim, step, mask_c, mask_h, hidden_tr, hidden_list, dropout):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, embedding_dim, step)
        for p in self.parameters():
            p.requires_grad = False
        self.mask_concat = MaskAndConcat(mask_c, mask_h, embedding_dim, hidden_tr)
        self.deep = DeepComponent(step*input_dim + hidden_tr, hidden_list, dropout)
        assert(step*input_dim + hidden_tr > hidden_list[-1])

    def forward(self, time_series):

        batch_size = time_series.size()[0]
        c_last = self.encoder(time_series)
        time_re = time_series.reshape(batch_size, -1)
        ori, con_inp, mask1, mask2 = self.mask_concat(time_re, c_last)
        deep_out = self.deep(con_inp)

        return deep_out, ori, mask1, mask2


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
embedding_dim = 24
step = 5
alpha = 0.5
belta = 0.5
batch_size = 512
hidden_tr = 12

mask_c = args.mask_rate
mask_h = args.mask_rate


mlp_list = (512, 256, 64)
dropout = 0
epochs = 50
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


name = str(args.mask_rate)+".log"
mylog = open(os.path.join(checkpoint_path, name), mode='a', encoding='utf-8')


ende_model = EncoderDecoderModel(input_dim, hidden_dim, embedding_dim, step, mask_c, mask_h, hidden_tr, mlp_list, dropout).to(device)
optimizer = Adam(filter(lambda p: p.requires_grad, ende_model.parameters()), lr=3e-4)

path = r"D:\Users\wt\Downloads\52_0.2692.pth.tar"
ende_model, optimizer = load_checkpoint(ende_model, path, optimizer, False)

criterion = RecLoss(df.std(), hidden_tr, step).to(device)
train_loader = DataLoader(PretrainDataset(train_left_des, step), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(PretrainDataset(val_left_des, step), batch_size=batch_size, shuffle=True)  # shuffle=False
best_loss = 999

save_dic = dict(train=[], val=[], batch=[])
for epoch in range(epochs):
    if epochs_since_improvement == 15:
        print("Reach epochs since improvement: save loss info!",)
        np.save(name[:-4]+".npy", save_dic)
        mylog.close()
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
        adjust_learning_rate(optimizer, 0.9)
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
                          loss=losses), file=mylog)
        save_dic['batch'].append(losses.avg)
    save_dic['train'].append(losses.avg)
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
                        'optimizer': optimizer.state_dict()}, os.path.join(checkpoint_path, name[:-4],
                                                                           str("%.4f.pth.tar" % best_loss)))
        else:
            epochs_since_improvement += 1
        print('Validation: [{0}/{1}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, val_samples,
                                                              batch_time=batch_time,
                                                              loss=val_losses), file=mylog)
    save_dic['val'].append(val_losses.avg)
print("Reach last epoch: save loss info!")
np.save(name[:-4]+".npy", save_dic)
mylog.close()