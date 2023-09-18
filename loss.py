import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from scipy import interpolate
from datetime import datetime
import pandas as pd
from tool import EarlyStopping
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error, r2_score
from common import *
from net import CRNN

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



import numpy as np



    



def loss_print(test_loader,rul_true,rul_pred)
    
    test_battery_ids = [i[0] for i in test_loader]
    test_channel = [i[1] for i in test_loader]
    cols = ['rmse_before','rmse_after',
       'r2_before','r2_after',
       'mae_before','mae_after',]
    table_wx = pd.DataFrame(index=[test_battery_ids,test_channel],columns=cols)
    table_wx.loc['All',:] = 0

    stride = 1
    rmse_after_list, r2_after_list, mae_after_list = [], [], []

    for code,battery_ids in test_loader[:]:
        # A = battery_ids
        # end = A['rul'][1]
        
        # A_rul = A['rul']
        # A_life = A['rul'][1]
        # rul_true = result[battery_ids]['rul']['true']
        # rul_base = result[battery_ids]['rul']['base']
        # rul_pred = result[battery_ids]['rul']['transfer']
        # start = end - len(rul_true)
    #     print(battery_ids, A_life, rul_true[0])

###################### MINE ###############################
        rul_true = rul_true
        rul_pred = rul_pred
        A_life = 
        
        rmse_after = np.sqrt(mean_squared_error(rul_true,rul_pred))
    #     print('rmse:','%.3g'%rmse_before,'%.3g'%rmse_after)
        table_wx.loc[(code, battery_ids),['rmse_after']] = ['%.3g'%rmse_after]
        
        r2_after = r2_score(rul_true,rul_pred)
    #     print('r2:','%.3g'%r2_before, '%.3g'%r2_after)
        table_wx.loc[(code, battery_ids),['r2_before','r2_after']] = [ '%.3g'%r2_after]
        
        mae_after = mean_absolute_error(rul_true, rul_pred) / A_life * 100
    #     print('mae/life:','%.3g'%mae_before, '%.3g'%mae_after)
        table_wx.loc[(code, battery_ids),['mae_before','mae_after']] = ['%.3g'%mae_after]
        
    #     print('')
        
        rmse_after_list.append(rmse_after)
        

        r2_after_list.append(r2_after)
        

        mae_after_list.append(mae_after)

    table_wx.loc['All',['rmse_before','rmse_after']] = [
                                                                    '%.3g'%np.mean(rmse_after_list)]
    table_wx.loc['All',['r2_before','r2_after']] = [
                                                            '%.3g'%np.mean(r2_after_list)]   
    table_wx.loc['All',['mae_before','mae_after']] = [
                                                                '%.3g'%np.mean(mae_after_list)]
