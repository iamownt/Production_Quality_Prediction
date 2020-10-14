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

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient
    dataloader tool provided by PyTorch.
    """

    def __init__(self, root, train=True):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.root = root
        self.train = train

        if not self._check_exists:
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train0515_4.csv')).iloc[1000:]  # 'train.txt'
            # data.iloc[:, -1] = (data.iloc[:, -1] - 2.24) / 1.11  # - 0.6)/(5.53-0.6)
            self.train_data = data.iloc[:, 1:8].values
            self.traindnn_data = data.iloc[:, 8:-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test0512.csv'))
            self.test_data = data.iloc[:, 1:8].values
            self.testdnn_data = data.iloc[:, 8:-1].values
            self.target = data.iloc[:, -1].values

    def __getitem__(self, idx):
        if self.train:
            dataI, dataD, targetI = self.train_data[idx, :], self.traindnn_data[idx, :], self.target[idx]
            Xi = torch.from_numpy(dataI.astype(np.long)).long()
            Xv = torch.from_numpy(np.ones_like(dataI))
            Xd = torch.from_numpy(dataD.astype(np.float32))  # .unsqueeze(-1)
            targetI = torch.Tensor([targetI])
            return Xi, Xv, targetI, Xd
        else:
            dataI, dataD, targetI = self.test_data[idx, :], self.testdnn_data[idx, :], self.target[idx]  ##.iloc
            Xi = torch.from_numpy(dataI.astype(np.long)).long()
            Xv = torch.from_numpy(np.ones_like(dataI))
            Xd = torch.from_numpy(dataD.astype(np.float32))  # .unsqueeze(-1)
            targetI = torch.Tensor([targetI])
            return Xi, Xv, targetI, Xd

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)

class MyMLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            # layers.append(torch.nn.LeakyReLU(0.8))
            # layers.append(torch.nn.PReLU(1, init=0.8))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = hidden_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)

class MyDeepFM(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, addition_input, addition_hiddens):
        super().__init__()
        # self.linear = FeaturesLinear(field_dims)
        # self.fm = FactorizationMachine(reduce_sum=True)
        # self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.embed_output_dim = len(field_dims) * embed_dim
        # self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.mlp2 = MyMLP(addition_input, addition_hiddens, dropout=dropout)

    def forward(self, x_sparse, x_dense):
        """
        :param x_dense: Long tensor of size ``(batch_size, num_fields)``
        :param x_sparse: for sparse vector with files
        """
        # embed_x = self.embedding(x_sparse)
        # part_1 = self.linear(x_sparse) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        part_2 = self.mlp2(x_dense)
        # x = part_1 + part_2
        x = part_2
        return x

    def rec(self,old):
        for i in range(len(old)):
            old[i] = (old[i] * 1.11) + 2.24
        return old

    def _initialize_weights(self):
        # print(self.modules())

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.001, std=0.001)
                print(m.weight, flush=True)


class SingleDeepFMDataset(Dataset):
    """
    Create for a simple DNN input
    默认df的最后一维度是target!
    """

    def __init__(self, df, split, last_item=10):

        self.split = split
        self.last_item = last_item
        self.df = df
        assert (self.split in {"train", "test"})
        # col_list = [col for col in self.df.columns if 'Flotation' not in col ]
        # if you want to select some col with col_name, use self.df = self.df[col_list]
        samples = int(len(self.df) / 180)
        if self.split == "train":
            self.df.iloc[:, -1] = (self.df.iloc[:, -1] - 2.24) / 1.11
            lis = [(list(range(180 + i * 180 - self.last_item, 180 + i * 180))) for i in range(samples)]
            ind = np.concatenate(lis)
            self.df = self.df.iloc[ind]
            self.dataset_size = int(len(self.df))
            print("Train: ", self.dataset_size)
        else:
            lis = [(list(range(179 + i * 180, 180 + i * 180))) for i in range(samples)]
            ind = np.concatenate(lis)
            self.df = self.df.iloc[ind]
            self.dataset_size = int(len(self.df))
            print("Test: ", self.dataset_size)

    def __getitem__(self, i):
        """
        :param i:
        :return:  (feature_size)
        """
        x_ori = torch.from_numpy(self.df.iloc[i, :-1].values).float().to(device)
        y_ori = torch.Tensor([self.df.iloc[i, -1]]).float().to(device)
        x_sparse = x_ori[23:].long()
        x_dense = x_ori[:23]
        return x_sparse, x_dense, y_ori

    def __len__(self):
        return self.dataset_size


# train_left_des = r"D:\Datasets\MiningProcessEngineering\train_left_now.csv"
# test_left_des = r"D:\Datasets\MiningProcessEngineering\test_left_now.csv"
# train_left_now = pd.read_csv(train_left_des)
# test_left_now = pd.read_csv(test_left_des)
# train_left_now.drop("date", axis=1, inplace=True)
# test_left_now.drop("date", axis=1, inplace=True)
#
# dense_features = [col for col in train_left_now.columns if "bin" not in col and col != "% Silica Concentrate"]
# sparse_features = [col for col in train_left_now.columns if col not in dense_features and col != "% Silica Concentrate"]
# train_left_now[sparse_features] = train_left_now[sparse_features].astype('long')
# test_left_now[sparse_features] = test_left_now[sparse_features].astype('long')


# 4 is the bin size
field_dims = [31,31,32,25,35,42,18]
embed_dim = 4
mlp_dims = (128, 128)
addition_hiddens = (256, 256, 32)
dropout = 0

batch_size = 150
epochs = 5000
print_freq = 600
count = 0

train_loss = []
test_loss = []

dfm_model = MyDeepFM(field_dims=field_dims, embed_dim=embed_dim, mlp_dims=mlp_dims, dropout=dropout,
                     addition_input=10, addition_hiddens=addition_hiddens)
dfm_model._initialize_weights()

criterion = nn.MSELoss().to(device)
optimizer = Adam(dfm_model.parameters(), lr=0.0003, weight_decay=1e-4)  # 3e-4 is recommend!
train_loader = DataLoader(CriteoDataset(r'C:\Users\wt\Desktop', train=True), batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(CriteoDataset(r'C:\Users\wt\Desktop', train=False), batch_size=744, shuffle=False)

# 我想做的是每个iteration进行训练，我觉得epoch去评估训练集，或者几个iteration去评估，但是eval得每个iter。
for epoch in range(epochs):
    # losses = AverageMeter()
    for i, (input_sparse, abee, label, input_dense) in enumerate(train_loader):
        count += 1
        dfm_model.train()
        target = dfm_model(input_sparse, input_dense)
        loss = criterion(target, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 这个losses是每个iteration的平均损失。
        # losses.update(loss.item())
        if i % print_freq == 0:
            print("Epoch: {0}, train mse：{1}".format(epoch, loss.item()))
        # train_loss.append(np.sqrt(losses.avg))
        if i % print_freq == 0:
            dfm_model.eval()
            with torch.no_grad():
                for _, (input_sparse, abee, label, input_dense) in enumerate(test_loader):
                    target = dfm_model(input_sparse, input_dense)
                    # recy = dfm_model.rec(target)
                    loss = criterion(target, label)
                    # test_loss.append(np.sqrt(loss.item()))
                    if i % print_freq == 0:
                        print("Epoch: {0}, iteration: {2}, test mse: {1}".format(epoch, loss.item(), count))