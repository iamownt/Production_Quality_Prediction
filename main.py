import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import MiningDataset, NeuralNetworkDataset
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, step, drop_out=0.5):
        super(LSTM, self).__init__()
        self.step = step
        self.dropout_rate = drop_out
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.decoder_step = nn.LSTMCell(input_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)

        return h, c

    def forward(self, time_series):
        """
        :param time_series: the expected time_series shape is (batch_size, time_step, feature_size)
        :return: the last time step's output followed by a mlp
        """
        batch_size = time_series.size(0)

        h, c = self.init_hidden_state(batch_size)

        for i in range(self.step):
            h, c = self.decoder_step(time_series[:, i, :], (h, c))
        out = self.fc1(c)
        #out = self.dropout(out)

        return out


class NN(nn.Module):

    def __init__(self, D_in, H1, H2, D_out, dropout=0.5):
        super(NN, self).__init__()
        self.dropout_rate = dropout
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)
        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)
        self.bn3 = nn.BatchNorm1d(D_out)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):

        x = self.relu1(self.bn1(self.fc1(x)))
        # x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        # x = self.dropout2(x)

        return x

if __name__ == '__main__':

    input_dim = 8
    hidden_dim = 512
    step = 5
    batch_size = 256
    epochs = 500
    grad_clip = 5.
    print_freq = 10
    epochs_since_improvement = 0
    output_folder = r"D:\Datasets\Mining_output"
    train_left_des = r"D:\Users\wt\Downloads\production_quality_prediction\train_left.csv"
    test_left_des = r"D:\Users\wt\Downloads\production_quality_prediction\test_left.csv"
    lstm_model = LSTM(input_dim, hidden_dim, step, 0).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = Adam(lstm_model.parameters(), lr=0.003)
    train_loader = DataLoader(MiningDataset(train_left_des), batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(MiningDataset(test_left_des), batch_size=744, shuffle=False, pin_memory=True)#670
    train_loss = []
    test_loss = []
    best_loss = 999
    for epoch in range(epochs):
        if epochs_since_improvement == 12:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)
        lstm_model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start = time.time()
        for i, (time_series, label) in enumerate(train_loader):
            data_time.update(time.time() - start)
            time_series.to(device)
            label.to(device)
            target = lstm_model(time_series)
            loss = criterion(target, label)
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
        train_loss.append(np.sqrt(losses.avg))

        # eval the model
        lstm_model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        start = time.time()
        with torch.no_grad():

            for i, (time_series, label) in enumerate(test_loader):
                time_series.to(device)
                label.to(device)
                label = label[-670:]
                target = lstm_model(time_series)[-670:]
                loss = criterion(target, label)

                losses.update(loss.item())
                batch_time.update(time.time() - start)

                start = time.time()
            test_rmse = np.sqrt(losses.avg)
            test_loss.append(test_rmse)
            if test_rmse < best_loss:
                best_loss = test_rmse
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(test_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))
    lstm_model.eval()

    prediction = lstm_model(time_series)

    plt.plot(prediction[-670:].detach().numpy(), label="prediction")
    plt.plot(label[-670:].detach().numpy(), label="label")
    plt.legend(loc="upper left")
    plt.show()
    plt.close()

    plt.plot(train_loss, label="Train")
    plt.plot(test_loss, label="Test")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
    print(train_loss)
    print(test_loss)
    print(best_loss)