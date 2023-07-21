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

def our_data_loader_generate():

    i_low = -2199
    i_upp = 5498
    v_low = 3.36
    v_upp = 3.60
    q_low = 610
    q_upp = 1190
    rul_factor = 3000
    cap_factor = 1190
    pkl_dir = './data/our_data/'
    series_lens = [100]

    new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
    new_train = ['9-1', '2-2', '4-7','9-7', '1-8','4-6','2-7','8-4', '7-2','10-3', '2-4', '7-4', '3-4',
            '5-4', '8-7','7-7', '4-4','1-3', '7-1','5-2', '6-4', '9-8','9-5','6-3','10-8','1-6','3-5',
             '2-6', '3-8', '3-6', '4-8', '7-8','5-1', '2-8', '8-2','1-5','7-3', '10-2','5-5', '9-2','5-6', '1-7', 
             '8-3', '4-1','4-2','1-4','6-5', ]
    new_test  = ['9-6','4-5','1-2', '10-7','1-1', '6-1','6-6', '9-4','10-4','8-5', '5-3','10-6',
            '2-5','6-2','3-1','8-8', '8-1','8-6','7-6','6-8','7-5','10-1']

    train_fea, train_lbl = [], []
    for name in new_train + new_valid:
        print(f"loading {name}")
        '''
        label: [rul, full_seq]
        '''
        tmp_fea, tmp_lbl = get_xy(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp,
                                                   rul_factor, cap_factor, pkl_dir, raw_features=False)
        # tmp_fea, tmp_lbl = get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
        train_fea.append(tmp_fea)
        train_lbl.append(tmp_lbl)
    
    test_fea, test_lbl = [], []
    for name in new_test:
        print(f"loading {name}")
        '''
        label: [rul, full_seq]
        '''
        tmp_fea, tmp_lbl = get_xy(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp,
                                                   rul_factor, cap_factor, pkl_dir, raw_features=False)
        # tmp_fea, tmp_lbl = get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
        test_fea.append(tmp_fea)
        test_lbl.append(tmp_lbl)

    
    train_fea = np.vstack(train_fea)
    train_lbl = np.vstack(train_lbl)
    test_fea = np.vstack(test_fea)
    test_lbl = np.vstack(test_lbl)

    train_soh_index = np.argsort(train_lbl[:, -1])
    sorted_train_fea = train_fea[train_soh_index]
    sorted_train_lbl = train_lbl[train_soh_index]
    # print(train_fea.shape, train_lbl.shape, test_fea.shape, test_lbl.shape)
    return sorted_train_fea, sorted_train_lbl, test_fea, test_lbl

if __name__ == "__main__":
    our_data_loader_generate()