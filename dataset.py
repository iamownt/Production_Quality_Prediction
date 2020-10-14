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
    def __init__(self, data_des, split, last_item=5):

        self.split = split
        assert(self.split in {"train", "test"})
        self.last_item = last_item
        self.df = pd.read_csv(data_des)
        self.df.drop("date", axis=1, inplace=True)
        col_list = [col for col in self.df.columns if 'Flotation' not in col ]
        self.df = self.df[col_list]

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
            data.iloc[:, -1] = (data.iloc[:, -1] - 2.24) / 1.11  # - 0.6)/(5.53-0.6)
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
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            Xd = torch.from_numpy(dataD.astype(np.float32))  # .unsqueeze(-1)
            return Xi, Xv, targetI, Xd
        else:
            dataI, dataD, targetI = self.test_data[idx, :], self.testdnn_data[idx, :], self.target[idx]  ##.iloc
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            Xd = torch.from_numpy(dataD.astype(np.float32))  # .unsqueeze(-1)

            return Xi, Xv, targetI, Xd

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)
