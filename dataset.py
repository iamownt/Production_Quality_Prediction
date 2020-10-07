import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MiningDataset(Dataset):
    """
    A Pytorch Dataset class to be used in PyTorch DataLoader to create batches
    """
    def __init__(self, data_des, last_item=10):

        self.last_item = last_item
        # self.df = pd.read_csv(data_des).iloc[:128*180,:]
        self.df = pd.read_csv(data_des)
        self.df.drop("date", axis=1, inplace=True)
        col_list = [col for col in self.df.columns if 'Flotation' not in col ]
        self.df = self.df[col_list]
        self.dataset_size = int(len(self.df)/180)

    def __getitem__(self, i):
        """
        :param i:
        :return:  (time_step. feature_size)
        """
        x_ori = torch.from_numpy(self.df.iloc[i*180:180 + i*180, :-1].values).float().to(device)
        y_ori = torch.Tensor([self.df.iloc[180 + i*180 - 1, -1]]).to(device)
        return x_ori[-self.last_item:], y_ori

    def __len__(self):
        return self.dataset_size


class NeuralNetworkDataset(Dataset):
    """
    Create for a simple DNN input
    """
    def __init__(self, data_des, split, last_item=10):

        self.split = split
        assert(self.split in {"train", "test"})
        self.last_item = last_item
        self.df = pd.read_csv(data_des)
        self.df.drop("date", axis=1, inplace=True)
        col_list = [col for col in self.df.columns if 'Flotation' not in col ]
        self.df = self.df[col_list]

        samples = int(len(self.df) / 180)
        if self.split == "train":
            lis = [(list(range(170 + i * 180, 180 + i * 180))) for i in range(samples)]
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
        #用最后一个时刻的
        # x_ori = torch.from_numpy(self.df.iloc[180 + i * 180 - 1, :22].values).float().to(device)
        # y_ori = torch.Tensor([self.df.iloc[180 + i * 180 - 1, 22]]).to(device)
        #用最后10个时刻的
        if self.split == 'train':
            x_ori = torch.from_numpy(self.df.iloc[i, :-1].values).float().to(device)
            y_ori = torch.Tensor([self.df.iloc[i, -1]]).float().to(device)
            return x_ori, y_ori

        else:
            x_ori = torch.from_numpy(self.df.iloc[i, :-1].values).float().to(device)
            y_ori = torch.Tensor([self.df.iloc[i, -1]]).float().to(device)
            return x_ori, y_ori

    def __len__(self):
        return self.dataset_size
