import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
import pandas as pd
from tool import EarlyStopping
from sklearn.metrics import roc_auc_score,mean_squared_error
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')


# save dict    
def save_obj(obj,name):
    with open(name + '.pkl','wb') as f:
        pickle.dump(obj,f)
                  
#load dict        
def load_obj(name):
    with open(name +'.pkl','rb') as f:
        return pickle.load(f)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def interp(v, q, num):
    f = interpolate.interp1d(v,q,kind='linear')
    v_new = np.linspace(v[0],v[-1],num)
    q_new = f(v_new)
    vq_new = np.concatenate((v_new.reshape(-1,1),q_new.reshape(-1,1)),axis=1)
    return q_new

def get_xy(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir,
             raw_features=True, fill_with_zero=True, seriesnum=1500):
    """
    Args:
        n_cyc (int): The previous cycles number for model input
        in_stride (int): The interval in the previous cycles number
        fea_num (int): The number of interpolation
        v_low (float): Voltage minimum for normalization
        v_upp (float): Voltage maximum for normalization
        q_low (float): Capacity minimum for normalization
        q_upp (float): Capacity maximum for normalization
        rul_factor (float): The RUL factor for normalization
        cap_factor (float): The capacity factor for normalization
        seriesnum: The number of series sliced from this degradation curve
    """
    def preprocess(data):
        datamean = np.mean(data)
        datastdvar = math.sqrt(np.var(data))
        return [datamean, datastdvar]
    # print("loading", name)
    if not os.path.exists(pkl_dir + name + '_fea.npy'):
        A = load_obj(pkl_dir + name)[name]
        A_rul = A['rul']
        A_dq = A['dq']
        A_df = A['data']
        all_fea = []
        all_idx = list(A_dq.keys())[9:]
        ruls = []
        for cyc in all_idx:
            feature = [A_dq[cyc] / cap_factor]
            time, v, i, q, dv, di, dq, dtime = [], [], [], [], [], [], [], []
            for timeidx in range(len(A_df[cyc]['Status'])):
                if 'discharge' in A_df[cyc]['Status'][timeidx]:
                    time.append(A_df[cyc]['Time (s)'][timeidx])
                    v.append((A_df[cyc]['Voltage (V)'][timeidx] - v_low) / (v_upp - v_low))
                    i.append((A_df[cyc]['Current (mA)'][timeidx] - i_low) / (i_upp - i_low))
                    q.append((A_df[cyc]['Capacity (mAh)'][timeidx] - q_low) / (q_upp - q_low))
                    if timeidx < len(A_df[all_idx[0]]['Voltage (V)']):
                        # import pdb;pdb.set_trace()
                        dv.append((A_df[cyc]['Voltage (V)'][timeidx] - A_df[all_idx[0]]['Voltage (V)'][timeidx]) / (v_upp - v_low))
                        di.append((A_df[cyc]['Current (mA)'][timeidx] - A_df[all_idx[0]]['Current (mA)'][timeidx]) / (i_upp - i_low))
                        dq.append((A_df[cyc]['Capacity (mAh)'][timeidx] - A_df[all_idx[0]]['Capacity (mAh)'][timeidx]) / (q_upp - q_low))
                        dtime.append(A_df[cyc]['Time (s)'][timeidx])
            feature += preprocess(v)
            feature += preprocess(i)
            feature += preprocess(q)
            feature += preprocess(dv)
            feature += preprocess(di)
            feature += preprocess(dq)
            all_fea.append(feature)
            ruls.append(A_rul[cyc])

        np.save(pkl_dir + name + '_fea.npy', all_fea)
        np.save(pkl_dir + name + '_rul.npy', ruls)
    else:
        all_fea = np.load(pkl_dir + name + '_fea.npy', allow_pickle=True)
        A_rul = np.load(pkl_dir + name + '_rul.npy', allow_pickle=True)
        # import pdb;pdb.set_trace()
    if raw_features:
        return np.array(all_fea), A_rul
    feature_num = len(all_fea[0])
    all_series, all_ruls = np.empty((0, np.max(series_lens), feature_num)), np.empty((0, 3))
    for ratio in range(4):
        tmpfea=copy.deepcopy(all_fea)
        tmprul=copy.deepcopy(A_rul)
        tmpfea=tmpfea[0::ratio+1]
        tmprul=tmprul[0::ratio+1]
        for series_len in series_lens:
            # series_num = len(all_fea) // series_len
            # series = np.lib.stride_tricks.as_strided(np.array(all_fea), (series_num, series_len, feature_num))
            series = np.lib.stride_tricks.sliding_window_view(tmpfea, (series_len, feature_num))
            series = series.squeeze()
            full_series = []
            if series_len < np.max(series_lens) and fill_with_zero:
                zeros = np.zeros((np.max(series_lens) - series_len, feature_num))
                for seriesidx in range(series.shape[0]):
                    # import pdb;pdb.set_trace()
                    full_series.append(np.concatenate((series[seriesidx], zeros)))
            elif series_len == np.max(series_lens):
                full_series = series
            # ruls = np.array(A_rul[series_len - 1:]) / rul_factor
            # series.tolist()
            full_series = np.array(full_series)

            full_seq_len = len(tmprul)

            if isinstance(A_rul, dict):
                tmp = []
                for k, v in A_rul.items():
                    if k >= series_len:
                        tmp.append([v / rul_factor, full_seq_len / rul_factor, v / full_seq_len])
                ruls = tmp
            else:
                ruls = tmprul[series_len/(ratio+1) - 1:].tolist()
                for i in range(len(ruls)):
                    ruls[i] = [ruls[i] / rul_factor, full_seq_len / rul_factor, ruls[i] / full_seq_len]
            # import pdb;pdb.set_trace()
            # print(all_series.shape, all_ruls.shape)
            all_series = np.append(all_series, full_series, axis=0)
            ruls = np.array(ruls).astype(float)
            all_ruls = np.append(all_ruls, ruls, axis=0)
    if seriesnum is not None:
        all_series = all_series[:seriesnum]
        all_ruls = all_ruls[:seriesnum]
    return all_series, all_ruls


class Trainer():
    
    def __init__(self, lr, n_epochs,device, patience, lamda, alpha, model_name):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss
            alpha (List: [float]): The weights of Capacity loss
            model_name (str): The model save path
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.lamda = lamda
        self.alpha = alpha

    def train(self, train_loader, valid_loader, model, load_model):
        model = model.to(self.device)
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=self.lr,)
        model_name = self.model_name
        lamda = self.lamda
        alpha = self.alpha
        
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(self.patience, verbose=True)
        loss_fn.to(self.device)

        # Training
        train_loss = []
        valid_loss = []
        total_loss = []
        
        for epoch in range(self.n_epochs):
            model.train()
            y_true, y_pred = [], []
            losses = []
            for step, (x,y) in enumerate(train_loader):  
                optimizer.zero_grad()
                
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                
                loss = lamda * loss_fn(y_.squeeze(), y[:,0])
                
                for i in range(y.shape[1] - 1):
                    loss += loss_fn(soh_[:,i], y[:,i+1]) * alpha[i]
                    
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())

                y_pred.append(y_)
                y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            train_loss.append(epoch_loss)
            
            losses = np.mean(losses)
            total_loss.append(losses)
            
            # validate
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for step, (x,y) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_, soh_ = model(x)

                    y_pred.append(y_)
                    y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            valid_loss.append(epoch_loss)
            
            if self.n_epochs > 100:
                if (epoch % 100 == 0 and epoch !=0):
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',f'-- "total" loss {losses:.4}')
            else :
                print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',f'-- "total" loss {losses:.4}')

            early_stopping(epoch_loss, model, f'{model_name}_best.pt')
            if early_stopping.early_stop:
                break
                
        if load_model:
            model.load_state_dict(torch.load(f'{model_name}_best.pt'))
        else:
            torch.save(model.state_dict(), f'{model_name}_end.pt')

        return model, train_loss, valid_loss, total_loss

    def test(self, test_loader, model):
        model = model.to(self.device)
        device = self.device

        y_true, y_pred, soh_true, soh_pred = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)

                y_pred.append(y_)
                y_true.append(y[:,0])
                soh_pred.append(soh_)
                soh_true.append(y[:,1:])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            soh_true = torch.cat(soh_true, axis=0)
            soh_pred = torch.cat(soh_pred, axis=0)
            mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return y_true, y_pred, mse_loss, soh_true, soh_pred
    

class FineTrainer():
    
    def __init__(self, lr, n_epochs,device, patience, lamda, train_alpha, valid_alpha, model_name):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss. In fine-tuning part, set 0.
            train_alpha (List: [float]): The weights of Capacity loss in model training
            valid_alpha (List: [float]): The weights of Capacity loss in model validation
            model_name (str): The model save path
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.lamda = lamda
        self.train_alpha = train_alpha
        self.valid_alpha = valid_alpha

    def train(self, train_loader, valid_loader, model, load_model):
        model = model.to(self.device)
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=self.lr,)
        model_name = self.model_name
        lamda = self.lamda
        train_alpha = self.train_alpha
        valid_alpha = self.valid_alpha
        
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(self.patience, verbose=True)
        loss_fn.to(self.device)

        # Training
        train_loss = []
        valid_loss = []
        total_loss = []
        added_loss = []
        
        for epoch in range(self.n_epochs):
            model.train()
            y_true, y_pred = [], []
            losses = []
            for step, (x,y) in enumerate(train_loader):  
                optimizer.zero_grad()
                
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                soh_ = soh_.view(y_.shape[0], -1)
                
                loss = lamda * loss_fn(y_.squeeze(), y[:,0])
                
                for i in range(y.shape[1] - 1):
                    loss += loss_fn(soh_[:,i], y[:,i+1]) * train_alpha[i]
                    
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())

                y_pred.append(y_)
                y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            train_loss.append(epoch_loss)
            
            losses = np.mean(losses)
            total_loss.append(losses)

            # validate
            model.eval()
            y_true, y_pred, all_true, all_pred = [], [], [], []
            with torch.no_grad():
                for step, (x,y) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_, soh_ = model(x)
                    soh_ = soh_.view(y_.shape[0], -1)

                    y_pred.append(y_)
                    y_true.append(y[:,0])
                    all_true.append(y[:,1:])
                    all_pred.append(soh_)

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            all_true = torch.cat(all_true, axis=0)
            all_pred = torch.cat(all_pred, axis=0)
            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            valid_loss.append(epoch_loss)
            
            temp = 0
            for i in range(all_true.shape[1]):
                temp += mean_squared_error(all_true[0:1,i].cpu().detach().numpy(), 
                                           all_pred[0:1,i].cpu().detach().numpy()) * valid_alpha[i]
            added_loss.append(temp)
            
            if self.n_epochs > 10:
                if (epoch % 200 == 0 and epoch !=0):
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',
                          f'-- "total" loss {losses:.4}',f'-- "added" loss {temp:.4}')
            else :
                print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',
                      f'-- "total" loss {losses:.4}',f'-- "added" loss {temp:.4}')

            early_stopping(temp, model, f'{model_name}_fine_best.pt')
            if early_stopping.early_stop:
                break
                
        if load_model:
            model.load_state_dict(torch.load(f'{model_name}_fine_best.pt'))
        else:
            torch.save(model.state_dict(), f'{model_name}_fine_end.pt')

        return model, train_loss, valid_loss, total_loss, added_loss

    def test(self, test_loader, model):
        model = model.to(self.device)
        device = self.device

        y_true, y_pred, soh_true, soh_pred = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                soh_ = soh_.view(y_.shape[0], -1)

                y_pred.append(y_)
                y_true.append(y[:,0])
                soh_pred.append(soh_)
                soh_true.append(y[:,1:])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            soh_true = torch.cat(soh_true, axis=0)
            soh_pred = torch.cat(soh_pred, axis=0)
            mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return y_true, y_pred, mse_loss, soh_true, soh_pred


def score_weight_loss(input, output):
    # 反转输入以使得较小的输入对应较大的输出
    inverted_input = 1.0 - input

    # 计算输入和输出的误差
    error = (inverted_input - output).abs()

    # 计算输入和输出间的差异
    diff_input = torch.abs(input[:-1] - input[1:])
    diff_output = torch.abs(output[:-1] - output[1:])

    # 计算差异的误差
    diff_error = (diff_input - diff_output).abs()

    # 将两个误差组合起来得到总的损失
    # loss = error.sum() + diff_error.sum()

    return error.sum(), diff_error.sum()
