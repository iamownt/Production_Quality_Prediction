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


class MyMLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.LeakyReLU(0.8))
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
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.step)])
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


class MyDeepFM(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, addition_input, addition_hiddens):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding1 = FeaturesEmbedding(field_dims, embed_dim)
        self.embedding2 = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.mlp2 = MyMLP(addition_input, addition_hiddens, dropout=dropout)
        self.t1 = torch.nn.Parameter(0.5*torch.ones(1))
        self.t2 = torch.nn.Parameter(0.5*torch.ones(1))

    def forward(self, x_sparse, x_dense):
        """
        :param x_dense: Long tensor of size ``(batch_size, num_fields)``
        :param x_sparse: for sparse vector with files
        """
        # embed_x1 = self.embedding1(x_sparse)
        # embed_x2 = self.embedding2(x_sparse)
        # part_1 = self.linear(x_sparse) + self.fm(embed_x1) + self.mlp(embed_x2.view(-1, self.embed_output_dim))
        part_2 = self.mlp2(x_dense)
        #x = self.t1 * part_1 + self.t2 * part_2
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
        self.df = self.df.iloc[:200*180, :] # TODO
        assert (self.split in {"train", "test"})
        # col_list = [col for col in self.df.columns if 'Flotation' not in col ]
        # if you want to select some col with col_name, use self.df = self.df[col_list]
        samples = int(len(self.df) / 180)
        if self.split == "train":
            # self.df.iloc[:, -1] = (self.df.iloc[:, -1] - 2.24) / 1.11
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


train_left_des = r"D:\Datasets\MiningProcessEngineering\train_left_now.csv"
test_left_des = r"D:\Datasets\MiningProcessEngineering\test_left_now.csv"
train_left_now = pd.read_csv(train_left_des)
test_left_now = pd.read_csv(test_left_des)
train_left_now.drop("date", axis=1, inplace=True)
test_left_now.drop("date", axis=1, inplace=True)

dense_features = [col for col in train_left_now.columns if "bin" not in col and col != "% Silica Concentrate"]
sparse_features = [col for col in train_left_now.columns if col not in dense_features and col != "% Silica Concentrate"]
train_left_now[sparse_features] = train_left_now[sparse_features].astype('long')
test_left_now[sparse_features] = test_left_now[sparse_features].astype('long')


# 4 is the bin size
field_dims = [4 for _ in range(len(sparse_features))]
embed_dim = 5
mlp_dims = (128, 128)
addition_hiddens = (256, 256, 32)
dropout = 0.5

batch_size = 12 #TODO
epochs = 150
print_freq = 600
count = 0

train_loss = []
test_loss = []

dfm_model = MyDeepFM(field_dims=field_dims, embed_dim=embed_dim, mlp_dims=mlp_dims, dropout=dropout,
                     addition_input=len(dense_features), addition_hiddens=addition_hiddens)
dfm_model._initialize_weights()

criterion = nn.MSELoss().to(device)
optimizer = Adam(dfm_model.parameters(), lr=3e-4, weight_decay=0.001)  # 3e-4 is recommend!
train_loader = DataLoader(SingleDeepFMDataset(train_left_now, "train", 2), batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(SingleDeepFMDataset(test_left_now, "test"), batch_size=744, shuffle=False)

# 我想做的是每个iteration进行训练，我觉得epoch去评估训练集，或者几个iteration去评估，但是eval得每个iter。
for epoch in range(epochs):
    # losses = AverageMeter()
    for i, (input_sparse, input_dense, label) in enumerate(train_loader):
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
                for _, (input_sparse, input_dense, label) in enumerate(test_loader):
                    label = label
                    target = dfm_model(input_sparse, input_dense)
                    # recy = dfm_model.rec(target)
                    loss = criterion(target, label)
                    # test_loss.append(np.sqrt(loss.item()))
                    if i % print_freq == 0:
                        print("Epoch: {0}, iteration: {2}, test mse: {1}".format(epoch, loss.item(), count))