import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from utils import *
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Save the training predictions
def ttt():
    path = r"D:\Datasets\Mining_output\checkpoint\34_0.5694.pth.tar"
    checkpoint = torch.load(path, map_location="cpu")
    train_loader = DataLoader(MMFineTuneDataset(train_left_des, step), batch_size=3342, shuffle=False)
    ende_model.load_state_dict(checkpoint['state_dict'])

    for tr, lb in train_loader:
        train_pred = ende_model(tr)
        print("Train RMSE: ", torch.sqrt(torch.mean(torch.square(train_pred - lb))))
        print(train_pred.size())

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


class MaskAndConcatFT(nn.Module):
    """
    注意torch.rand放cuda里面。
    mask1.to(device)，实际上无效，得设置mask1=mask1.to(device)， 但是对于模型来说，只用model.eval()就相当于可以了，不用model=model.eval()
    """
    def __init__(self, tr_in, tr_out):
        super(MaskAndConcatFT, self).__init__()
        self.fc = nn.Linear(tr_in, tr_out)

    def forward(self, con_in, h_in):
        # con_in(B, step*D)
        h_tr = self.fc(h_in)  # (B, tr_out)
        h_tr = (h_tr - torch.mean(h_tr, dim=0))/torch.std(h_tr, dim=0)
        ori = torch.cat([con_in, h_tr], dim=1)
        return ori


class DeepPart1(nn.Module):

    """
    注意模型的输入维度是shape(con_in) + shape(h_tr)， 不能小于中间层，
    """
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=False):
        super(DeepPart1, self).__init__()
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


class DeepPart2(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super(DeepPart2, self).__init__()
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


class EncoderDecoderModelFT(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, step, hidden_tr, hidden_list, hidden_list2, dropout):
        super(EncoderDecoderModelFT, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, embedding_dim, step)
        self.mask_concat = MaskAndConcatFT(embedding_dim, hidden_tr)
        self.deep = DeepPart1(step*input_dim + hidden_tr, hidden_list, dropout)
        for p in self.parameters():
            p.requires_grad = False
        self.deep2 = DeepPart2(hidden_list[-1], hidden_list2, dropout)

    def forward(self, time_series):

        batch_size = time_series.size()[0]
        c_last = self.encoder(time_series)
        time_re = time_series.reshape(batch_size, -1)
        ori = self.mask_concat(time_re, c_last)
        deep_out = self.deep(ori)
        # Below is the fine tune part
        deep_out = self.deep2(deep_out)

        return deep_out


class MMFineTuneDataset(Dataset):
    """
    A Pytorch Dataset class to be used in PyTorch DataLoader to create batches
    """
    def __init__(self, data_des, last_item=5, split="train"):
        assert split in ['train', 'val']

        self.last_item = last_item
        self.df = pd.read_csv(data_des) # 一共3342个样本
        if split == 'train':
            self.df = pd.read_csv(data_des).iloc[-N*180:, :]
        col_leave = ['% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
                     'Flotation Column 07 Air Flow', 'Flotation Column 03 Level', 'Flotation Column 07 Level',
                     '% Silica Concentrate', '2_hour_delay', '3_hour_delay']
        col_list = [col for col in self.df.columns if col in col_leave]
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
#N_set = list(np.linspace(10, 100, 10).astype("int"))+list(np.linspace(200, 3300, 32))
N_set=[3342]
NN = 1
typess = [2, 3]
record_train = defaultdict(list)
record_val = defaultdict(list)
for i in range(len(N_set)):
    N = int(N_set[i])
    for jjj in [1]:
        bugfight = jjj%3
        batch_size = 256

        epochs = 56 # 56
        mlp_list2 = (128, 64)
        learning_rate = 0.03

        hidden_tr = 12
        mlp_list = (512, 256, 64)
        # mlp_list2 = (12, 1)  # for finetune
        dropout = 0
        grad_clip = 5.
        print_freq = 1
        epochs_since_improvement = 0
        output_folder = r"D:\Datasets\Mining_output"
        checkpoint_path = r"C:\Users\wt\Desktop\experimente_features"
        train_left_des = r"D:\Datasets\MiningProcessEngineering\train_left_now.csv"
        val_left_des = r"D:\Datasets\MiningProcessEngineering\test_left_now.csv"
        df = pd.read_csv(train_left_des)
        col_leave = ['% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
                     'Flotation Column 07 Air Flow', 'Flotation Column 03 Level', 'Flotation Column 07 Level',
                     '2_hour_delay', '3_hour_delay']
        col_list = [col for col in df.columns if col in col_leave]
        df = df[col_list]

        ende_model = EncoderDecoderModelFT(input_dim, hidden_dim, embedding_dim, step, hidden_tr, mlp_list, mlp_list2, dropout).to(device)
        optimizer = Adam(filter(lambda p: p.requires_grad, ende_model.parameters()), lr=learning_rate) # TODO 0.003
        # optimizer = SGD(filter(lambda p: p.requires_grad, ende_model.parameters()), lr=3e-4, momentum=0.9)

        # ttt()
        if bugfight==0:
            wdc='meiyou'
            print(bugfight)
        elif bugfight==1:
            path = r"D:\Users\wt\Downloads\52_0.2692.pth.tar"
            ende_model, optimizer = load_checkpoint(ende_model, path, optimizer, False)
            ende_model.to(device)
            wdc="typeo"
            print(bugfight)
        elif bugfight==2:
            path = r"D:\Users\wt\Downloads\pretrained_ende\0.1390.pth.tar"
            ende_model, optimizer = load_checkpoint(ende_model, path, optimizer, False)
            ende_model.to(device)
            wdc = "typet"
            print(bugfight)

        # path = r"D:\Users\wt\Downloads\pretrained_ende\0.2189.pth.tar"  # TODO 最好是0.1390
        # path = r"D:\Users\wt\Downloads\pretrained_ende\0.1390.pth.tar"
        # ende_model, optimizer = load_checkpoint(ende_model, path, optimizer, False)


        # path = r"D:\Users\wt\Downloads\52_0.2692.pth.tar"
        # model, optimizer = load_checkpoint(ende_model, path, optimizer, False)


        criterion = nn.MSELoss().to(device)
        # criterion = nn.L1Loss().to(device)
        train_loader = DataLoader(MMFineTuneDataset(train_left_des, step, 'train'), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(MMFineTuneDataset(val_left_des, step, 'val'), batch_size=744, shuffle=False)  # shuffle=False
        num_cycle = 6 # epochs/8
        scheduler = CosineAnnealingWarmRestarts(optimizer, eta_min=1e-6, T_0=1000, T_mult=2)  # T_0=len(train_loader)*num_cycle

        best_loss = 999
        history = dict(train=[], val=[], batch=[])
        for epoch in range(epochs):
            # if epochs_since_improvement == 50:
                # plt.plot(history['train'][:10])
                # plt.plot(history['val'][10:])
                # plt.title('Model loss')
                # plt.ylabel('Loss')
                # plt.xlabel('Epoch')
                # plt.legend(['Train', 'Val'], loc='upper left')
                # plt.show()
            # if epochs_since_improvement > 0 and epochs_since_improvement % 50 == 0:
            #     adjust_learning_rate(optimizer, 0.9)
            ende_model.train()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            start = time.time()
            for i, (time_series, label) in enumerate(train_loader):
                data_time.update(time.time() - start)
                time_series.to(device)
                output = ende_model(time_series)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    clip_gradient(optimizer, grad_clip)
                optimizer.step()
                scheduler.step()

                losses.update(loss.item())
                batch_time.update(time.time() - start)
                # if i > print_freq:
                #     break

                start = time.time()
                # print status
                # if i % print_freq == 0:
                history["batch"].append(losses.avg)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses))
            history["train"].append(losses.avg)

            if epoch >= 5:
                # eval the model
                ende_model.eval()
                batch_time = AverageMeter()
                val_losses = AverageMeter()
                start = time.time()
                with torch.no_grad():

                    for tt, (time_series, label) in enumerate(val_loader):
                        time_series.to(device)
                        output = ende_model(time_series)
                        loss = criterion(output, label)
                        val_losses.update(loss.item())
                        batch_time.update(time.time() - start)

                        start = time.time()

                        # if i >= val_samples:
                        #     break
                    val_mse = val_losses.avg
                    history['val'].append(val_mse)
                    if val_mse < best_loss:
                        best_loss = val_mse
                        np.save(r"C:\Users\wt\Desktop\outputviz\v_2\val_preds.npy", output.detach().numpy())
                        np.save(r"C:\Users\wt\Desktop\outputviz\v_2\val_labels.npy", label.detach().numpy())
                        epochs_since_improvement = 0
                        if val_mse <= 0.57:
                            torch.save({'epoch': epoch+1, 'state_dict': ende_model.state_dict(), 'best_loss': best_loss,
                                        'optimizer': optimizer.state_dict()}, os.path.join(checkpoint_path,
                                                                                           str("%d_%.4f.pth.tar" % (epoch, val_mse))))
                            np.save(r"C:\Users\wt\Desktop\outputviz\v_2\abc"+str(val_mse)+".npy", output.detach().numpy())
                            # np.save(r"C:\Users\wt\Desktop\outputviz\v_2\abl.npy", label.detach().numpy())
                    else:
                        epochs_since_improvement += 1
                    print('Validation: [{0}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(tt,
                                                                          batch_time=batch_time,
                                                                          loss=val_losses))

        print("Reach last epoch: save loss info!")
        if bugfight == 0:
            np.save(r"C:\Users\wt\Desktop\1029"+ "\\type1_"+str(N) + ".npy", history)
        elif bugfight == 1:
            np.save(r"C:\Users\wt\Desktop\1029" + "\\type2_" + str(N) + ".npy", history)
        else:
            np.save(r"C:\Users\wt\Desktop\1029" + "\\type3_" + str(N) + ".npy", history)
        print("Min train mse: {0} Min val mse: {1}".format(min(history['train']), min(history['val'])))
        record_train[str(N)].append(history['train'])
        record_val[str(N)].append(history['val'])

np.save("record_tr.npy", record_train)
np.save("record_val.npy", record_val)