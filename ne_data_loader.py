import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy import interpolate
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error, r2_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import get_xy

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')


def ne_data_loader_generate(pkl_dir='./data/ne_data/'):

    i_low = -2199
    i_upp = 5498
    v_low = 3.36
    v_upp = 3.60
    q_low = 610
    q_upp = 1190
    rul_factor = 3000
    cap_factor = 1190
    pkl_dir = pkl_dir
    series_lens = [100]


    ne_valid = ['c0', 'c1', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10',
        'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19',
        'c20', 'c21', 'c22', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29',
        'c30', 'c31', 'c33', 'c34', 'c35', 'c36', 'c40', 'c41', 'c42',
        'c43', 'c44', 'c45']
    ne_train = ['a1', 'a3', 'a5', 'a7', 'a11', 'a15', 'a17', 'a19', 'a21', 'a24',
        'a26', 'a28', 'a30', 'a32', 'a34', 'a36', 'a38', 'a40', 'a42',
        'a44', 'b0', 'b2', 'b4', 'b6', 'b11', 'b13', 'b17', 'b19', 'b21',
        'b23', 'b25', 'b27', 'b29', 'b31', 'b33', 'b35', 'b37', 'b39',
        'b41', 'b43', 'b45']
    ne_test = ['a0', 'a2', 'a4', 'a6', 'a9', 'a14', 'a16', 'a18', 'a20', 'a23',
        'a25', 'a27', 'a29', 'a31', 'a33', 'a35', 'a37', 'a39', 'a41',
        'a43', 'a45', 'b1', 'b3', 'b5', 'b10', 'b12', 'b14', 'b18',
        'b20', 'b22', 'b24', 'b26', 'b28', 'b30', 'b32', 'b34', 'b36',
        'b38', 'b40', 'b42', 'b44', 'b46', 'b47']

    train_fea, train_lbl = [], []
    allseqs, allruls, batteryids = [], [], []
    batteryid = 0
    for name in ne_valid + ne_train:
        print(f"loading {name}")
        seqname = './data/ne_data/' + name + '.npy'
        lblname = './data/ne_data/' + name + '_rul.npy'
        seq = np.load(seqname, allow_pickle=True)
        lbls = np.load(lblname, allow_pickle=True)
        lbls = lbls / rul_factor
        tmp_fea,tmp_lbl = seq,lbls
        train_fea.append(tmp_fea)
        train_lbl.append(tmp_lbl)

    test_fea, test_lbl = [], []
    for name in ne_test:
        print(f"loading {name}")
        seqname = './data/ne_data/' + name + '.npy'
        lblname = './data/ne_data/' + name + '_rul.npy'
        seq = np.load(seqname, allow_pickle=True)
        lbls = np.load(lblname, allow_pickle=True)
        lbls = lbls / rul_factor
        tmp_fea,tmp_lbl = seq,lbls
        test_fea.append(tmp_fea)
        test_lbl.append(tmp_lbl)

    return train_fea, train_lbl, test_fea, test_lbl
        




class NEBatteryDataset(Dataset):
    def __init__(self, data_dir='./data/ne_data/', train=True, seqlen=100, stride=10):
        train_fea, train_lbl, test_fea, test_lbl = ne_data_loader_generate(data_dir)
        self.seqlen = seqlen
        self.stride = stride
        if train:
            self.fea = train_fea
            self.lbl = train_lbl
        else:
            self.fea = test_fea
            self.lbl = test_lbl
        if train:
            self.samples = self._gen_samples()
        else:
            self.samples=self._gen_samples_test()
    def _gen_samples(self):
        samples = []
        for battery_id, fea in enumerate(self.fea):
            lbl = self.lbl[battery_id]
            print(f'battery {battery_id}, num_cycles {len(fea)}, rul@0 {lbl[0]}, last_cycle_SOH {fea[-1][0]}')
            for cycle_id in list(range(len(fea)))[10::self.stride]:
                if cycle_id + self.seqlen >= len(fea) - 1 or cycle_id + self.seqlen >=1500 :
                    break
                fea_seq = fea[cycle_id:cycle_id+self.seqlen]
                rul = lbl[cycle_id+self.seqlen]
                samples.append((
                    torch.tensor(battery_id, dtype=torch.int),
                    torch.tensor(fea_seq, dtype=torch.float),
                    torch.tensor(rul, dtype=torch.float)))
        return samples
    def _gen_samples_test(self):
        samples = []
        for battery_id, fea in enumerate(self.fea):
            lbl = self.lbl[battery_id]
            print(f'battery {battery_id}, num_cycles {len(fea)}, rul@0 {lbl[0]}, last_cycle_SOH {fea[-1][0]}')
            for cycle_id in list(range(len(fea)))[10::1]:
                if cycle_id + self.seqlen >= len(fea) - 1 or cycle_id + self.seqlen >=1500 :
                    break
                fea_seq = fea[cycle_id:cycle_id+self.seqlen]
                rul = lbl[cycle_id+self.seqlen]
                samples.append((
                    torch.tensor(battery_id, dtype=torch.int),
                    torch.tensor(fea_seq, dtype=torch.float),
                    torch.tensor(rul, dtype=torch.float)))
        return samples        
    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    ne_data_loader_generate()