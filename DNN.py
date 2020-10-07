import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import MiningDataset, NeuralNetworkDataset
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)


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

        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = self.relu3(x)
        # x = self.dropout2(x)

        return x


dnn_input = 8
batch_size = 1024
epochs = 40
print_freq = 10
epochs_since_improvement = 0
output_folder = r"D:\Datasets\Mining_output"
train_left_des = r"D:\Users\wt\Downloads\production_quality_prediction\train_left.csv"
test_left_des = r"D:\Users\wt\Downloads\production_quality_prediction\test_left.csv"
dnn_model = NN(dnn_input, 256, 128, 1)
criterion = nn.MSELoss().to(device)
optimizer = Adam(dnn_model.parameters(), lr=3e-4)
train_loader = DataLoader(NeuralNetworkDataset(train_left_des, "train"), batch_size=batch_size, shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(NeuralNetworkDataset(test_left_des, "test"), batch_size=744, shuffle=False,
                         pin_memory=True)
train_loss = []
test_loss = []
best_loss = 999
for epoch in range(epochs):
    if epochs_since_improvement == 15:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
        adjust_learning_rate(optimizer, 0.8)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    dnn_model.train()
    for i, (time_series, label) in enumerate(train_loader):
        data_time.update(time.time() - start)
        time_series.to(device)
        label.to(device)
        label = label.reshape(-1, 1)

        target = dnn_model(time_series)
        loss = criterion(target, label)
        optimizer.zero_grad()
        loss.backward()
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
    dnn_model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    with torch.no_grad():

        for i, (time_series, label) in enumerate(test_loader):
            time_series.to(device)
            label.to(device)
            label = label.reshape(-1, 1)[-670:]
            target = dnn_model(time_series)[-670:]
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

dnn_model.eval()
prediction = dnn_model(time_series)[-670:]
np.save(os.path.join(output_folder, 'prediction.npy'), prediction.detach().numpy())

plt.plot(prediction.detach().numpy(), label="prediction")
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

# from sklearn.metrics import mean_squared_error
# import numpy as np
#
# lis = [i*180+180-1 for i in range(670)]
# df = pd.read_csv(r"D:\Users\wt\Downloads\production_quality_prediction\test_left.csv")
# df.drop("date", axis=1, inplace=True)
# out = np.load(r"D:\Datasets\Mining_output\prediction.npy")
# print(np.sqrt(mean_squared_error(df.iloc[lis, -1].values.reshape(-1, 1), out)))
