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
from matplotlib.ticker import StrMethodFormatter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)


class NN(nn.Module):

    def __init__(self, D_in, H1, H2, D_out, dropout=0):
        super(NN, self).__init__()
        self.dropout_rate = dropout
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)
        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)
        self.relu1 = nn.LeakyReLU(0.8)
        self.relu2 = nn.LeakyReLU(0.8)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # x = self.dropout2(x)

        return x


class LossI(nn.Module):

    def __init__(self, epsilon1, epsilon2, delta):
        super(LossI, self).__init__()
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.delta = delta

    def forward(self, preds, labels):
        error = preds - labels
        mask1 = (labels >= self.epsilon2)
        mask2 = (labels <= self.epsilon1)
        mask = mask1 + mask2
        lossi = (1 / preds.size()[0]) * (self.delta * torch.sum(torch.square(mask*error)) +
                                         (1 - self.delta) * torch.sum(torch.square(~mask*error)))
        return lossi


dnn_input = 9
batch_size = 256
epochs = 100
print_freq = 100
epochs_since_improvement = 0
output_folder = r"C:\Users\wt\Desktop\outputviz"
train_left_des = r"D:\Users\wt\Downloads\production_quality_prediction\train_left.csv"
test_left_des = r"D:\Users\wt\Downloads\production_quality_prediction\test_left.csv"
dnn_model = NN(dnn_input, 256, 128, 1)
criterion = nn.MSELoss().to(device)
#criterion = LossI(2, 4, 0.95)
optimizer = Adam(dnn_model.parameters(), lr=3e-3)
train_loader = DataLoader(NeuralNetworkDataset(train_left_des, "train", 1), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(NeuralNetworkDataset(test_left_des, "test"), batch_size=744, shuffle=False)
train_loss = []
test_loss = []
best_loss = 999
for epoch in range(epochs):
    if epochs_since_improvement == 10:
        print("Epochs since improvement: {0}. Stop training!\n Best Loss is {1}".format(epochs_since_improvement, best_loss))
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
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
            label = label.reshape(-1, 1)
            target = dnn_model(time_series)
            loss = criterion(target, label)

            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()
        test_rmse = np.sqrt(losses.avg)
        test_loss.append(test_rmse)
        if test_rmse < best_loss:
            np.save(os.path.join(output_folder, 'preds_'+'best'+'.npy'), target[:].detach().numpy())
            best_loss = test_rmse
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        print('Validation: [{0}/{1}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(test_loader),
                                                              batch_time=batch_time,
                                                              loss=losses))
        np.save(os.path.join(output_folder, 'labels.npy'), label[:].detach().numpy())



        # prediction = dnn_model(time_series)[-670:]
        # # np.save(os.path.join(output_folder, 'prediction.npy'), prediction.detach().numpy())
        #
        # plt.plot(prediction.detach().numpy(), label="prediction")
        # plt.plot(label[-670:].detach().numpy(), label="label")
        # plt.legend(loc="upper left")
        # plt.show()
        # plt.close()

    #
    # if epoch > 15 and epoch % 5 ==0:
    #     dnn_model.eval()
    #     prediction = dnn_model(time_series)[-670:]
    #     # np.save(os.path.join(output_folder, 'prediction.npy'), prediction.detach().numpy())
    #
    #     plt.plot(prediction.detach().numpy(), label="prediction")
    #     plt.plot(label[-670:].detach().numpy(), label="label")
    #     plt.legend(loc="upper left")
    #     plt.show()
    #     plt.close()
    #
    #     plt.plot(train_loss, label="Train")
    #     plt.plot(test_loss, label="Test")
    #     plt.title('Model loss')
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(loc='upper left')
    #     plt.show()
    #     print(train_loss)
    #     print(test_loss)
    #     print(best_loss)

# from sklearn.metrics import mean_squared_error
# import numpy as np
#
# lis = [i*180+180-1 for i in range(670)]
# df = pd.read_csv(r"D:\Users\wt\Downloads\production_quality_prediction\test_left.csv")
# df.drop("date", axis=1, inplace=True)
# out = np.load(r"D:\Datasets\Mining_output\prediction.npy")
# print(np.sqrt(mean_squared_error(df.iloc[lis, -1].values.reshape(-1, 1), out)))
#
# preds = np.load(r"C:\Users\wt\Desktop\outputviz\preds_768.npy")
# labels = np.load(r"C:\Users\wt\Desktop\outputviz\labels.npy")
# do_some_viz(preds, labels, r"C:\Users\wt\Desktop\outputviz", 'loss_ori')