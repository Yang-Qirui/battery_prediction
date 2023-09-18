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

def nmc_data_loader_generate(pkl_dir='./data/NMC_data/'):
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


    nmc_train = ['b','c','d','g','j','l','n','o','p','t']
    nmc_test  = ['a','e','f','s']

    train_fea, train_lbl = [], []
    for name in nmc_train:
        print(f"loading {name}")
        seqname = './data/NMC_data/' + name + '_fea.npy'
        lblname = './data/NMC_data/' + name + '_rul.npy'
        seq = np.load(seqname, allow_pickle=True)
        lbls = np.load(lblname, allow_pickle=True)
        lbls = lbls / rul_factor
        tmp_fea,tmp_lbl = seq,lbls
        train_fea.append(tmp_fea)
        train_lbl.append(tmp_lbl)
    
    test_fea, test_lbl = [], []
    for name in nmc_test:
        print(f"loading {name}")
        seqname = './data/NMC_data/' + name + '_fea.npy'
        lblname = './data/NMC_data/' + name + '_rul.npy'
        seq = np.load(seqname, allow_pickle=True)
        lbls = np.load(lblname, allow_pickle=True)
        lbls = lbls / rul_factor
        tmp_fea,tmp_lbl = seq,lbls
        test_fea.append(tmp_fea)
        test_lbl.append(tmp_lbl)

    return train_fea, train_lbl, test_fea, test_lbl

class NMC_BatteryDataset(Dataset):
    def __init__(self, data_dir='./data/NMC_data/',train=True, seqlen=100, stride=10):
        train_fea, train_lbl, test_fea, test_lbl = nmc_data_loader_generate()
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
    nmc_data_loader_generate()