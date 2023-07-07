from data_aug import data_aug, interpolate
import numpy as np
import argparse
from scipy.optimize import root
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SeqDataset
from dtw import *
from mlp import MLP
import matplotlib.pyplot as plt
import os

import wandb
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
import dataloader

class NegativeMSELoss(nn.Module):
    def __init__(self):
        super(NegativeMSELoss, self).__init__()

    def forward(self, input, target):
        return -torch.mean((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

# def custom_loss(y_pred, x):
#     mean_centered_pred = y_pred - torch.mean(y_pred)
#     mean_centered_input = x - torch.mean(x)
#     corr = torch.sum(mean_centered_pred * mean_centered_input) / (torch.sqrt(torch.sum(mean_centered_pred ** 2)) * torch.sqrt(torch.sum(mean_centered_input ** 2)))
#     return corr

def custom_loss(input, output):
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

def build_retrieval_set(curve_funcs, curve_lens, soh, seq_len):
    retrieval_set = []
    for j, curve_func in enumerate(curve_funcs):
        point = root(get_root, x0=0, args=(curve_func, soh))
        curve_len = curve_lens[j]
        start_cycle = round(point.x.item() * curve_len)
        retrieval_seq = []
        for k in range(seq_len):
            retrieval_seq.append(curve_func((start_cycle - seq_len + k) / curve_len))
        retrieval_set.append(np.array(retrieval_seq))
    return retrieval_set

def build_dataset(raw, seq_len, N):
    seqs = []
    targets = []
    for i in range(raw.shape[0] - seq_len - N):
        seq = raw[i: i + seq_len]
        target = raw[i + seq_len + N - 1]
        seqs.append(seq)
        targets.append([N, *target])
    return np.array(seqs), np.array(targets)

def get_root(x, spline, y_target):
    return spline(x) - y_target 

def featuremodel():

    return 

def data_process():

    n_cyc = 30
    in_stride = 3
    fea_num = 100

    v_low = 3.36
    v_upp = 3.60
    q_low = 610
    q_upp = 1190
    rul_factor = 3000.
    cap_factor = 1190
    i_low = -2199
    i_upp = 5498
    pkl_dir = './our_data/'
    pkl_list = os.listdir(pkl_dir)

    # new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
    new_train = ['9-1', '2-2', '4-7', '9-7', '1-8', '4-6', '2-7', '8-4', '7-2', '10-3', '2-4', '7-4', '3-4',
                 '5-4', '8-7', '7-7', '4-4', '1-3', '7-1', '5-2', '6-4', '9-8', '9-5', '6-3', '10-8', '1-6', '3-5',
                 '2-6', '3-8', '3-6', '4-8', '7-8', '5-1', '2-8', '8-2', '1-5', '7-3', '10-2', '5-5', '9-2', '5-6', '1-7',
                 '8-3', '4-1', '4-2', '1-4']
    new_test = ['9-6', '4-5', '1-2', '10-7', '1-1', '6-1', '6-6', '9-4', '10-4', '8-5', '5-3', '10-6',
                '2-5', '6-2', '3-1', '8-8', '8-1', '8-6', '7-6', '6-8', '7-5', '10-1',
                '4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7', '6-5']

    train_fea, train_ruls, train_batteryids = [], [], []
    seq_len = 100
    series_lens = [100]

    batch_size = 32
    valida_batch_size = 1
    seriesnum=1500
    scale_ratios = [1, 2, 3, 4] # [1, 2, 3]  # must in ascent order, e.g. [1, 2, 3]
    except_ratios = [[1, 2], [2, 1],
                 [2, 2],
                 [1, 3], [3, 1], [3, 3],
                 [1, 4], [2, 4], [4, 1], [4, 2], [4, 4]]
    parts_num_per_ratio=240
    valid_max_len = 10
    # wandb.init(project='battery_rul_predict',
    #         config={
    #                 'batch_size': 32,
    #                 'valida_batch_size': 1,
    #                 'seriesnum':1500,
    #                 'scale_ratios' : '[1, 2, 3]', # [1, 2, 3]  # must in ascent order, e.g. [1, 2, 3]
    #                 'except_ratios' : '[[1, 2], [2, 1], [2, 2], [1, 3], [3, 1], [3, 3], [1, 4], [2, 4], [4, 1], [4, 2], [4, 4]]',
    #                 'parts_num_per_ratio':240,
    #                 'valid_max_len': 10
    #             }
    #         )
    train_fea, train_rul, train_ruls, train_batteryids = [], [], [], []
    batteryid = 0
    for name in new_train:
        # tmp_fea, tmp_lbl = dataloader.get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
        tmp_fea, tmp_lbl = dataloader.get_xyv2(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir, raw_features=False, seriesnum=seriesnum)
        train_fea.append(tmp_fea)
        train_ruls.append(tmp_lbl)
        train_batteryids += [batteryid for _ in range(tmp_fea.shape[0])]
        batteryid += 1

    retrieval_set = {}
    batteryid = 0
    for name in new_train:
        retrieval_set[batteryid] = dataloader.get_retrieval_seq(name, pkl_dir, rul_factor, seriesnum=seriesnum)
        batteryid += 1

    

    train_fea = np.vstack(train_fea)
    train_ruls = np.vstack(train_ruls)
    train_rul = train_ruls[:,0]
    # import pdb;pdb.set_trace()
    train_batteryids = np.array(train_batteryids)
    train_batteryids = train_batteryids.reshape((-1, 1))
    # train_lbl = np.hstack((train_rul, train_batteryids))

    valid_fea, valid_rul, valid_ruls, valid_batteryids = [], [], [], []
    valid_battery_id = 0

    for name in new_test:
        tmp_fea, tmp_lbl = dataloader.get_xyv2(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir, raw_features=False)
        valid_fea.append(tmp_fea[:valid_max_len])#[::stride])
        valid_ruls.append(tmp_lbl[:valid_max_len])#[::stride])strid
        valid_batteryids += [valid_battery_id for i in range(len(tmp_fea))][:valid_max_len]#[::e]))]
        valid_battery_id += 1
    valid_fea = np.vstack(valid_fea)
    valid_ruls = np.vstack(valid_ruls)#.squeeze()
    valid_rul = valid_ruls[:,0]
    valid_batteryids = np.array(valid_batteryids)
    valid_batteryids = valid_batteryids.reshape((-1, 1))
    # valid_lbl = np.hstack((valid_rul, valid_batteryids))
    # print(train_fea.shape,  valid_fea.shape)
    # print(train_rul.shape,train_batteryids.shape,valid_rul.shape,valid_batteryids.shape)
    # print(train_rul)

    # trainset = TensorDataset(torch.Tensor(train_fea), torch.Tensor(train_lbl))
    # validset = TensorDataset(torch.Tensor(valid_fea), torch.Tensor(valid_lbl))

    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(validset, batch_size=valida_batch_size)


    return train_fea, train_rul, train_batteryids,valid_fea, valid_rul,valid_batteryids




def run(seq_len, N, curve_lens, curve_funcs, args):
    # train_data = np.load('./dataset/train/soh.npy')
    # test_data = np.load('./dataset/test/soh.npy')
    train_fea,train_data,train_batteryids,test_fea,test_data,valid_batteryids = data_process()
    _, smooth_train_data, _, _, _, _ = interpolate(train_data)
    _, smooth_test_data, _, _, _, _ = interpolate(test_data)
    smooth_train_data = smooth_train_data.reshape(-1, 1)
    smooth_test_data = smooth_test_data.reshape(-1, 1)
    # for i, curve in enumerate(curves):
        # plt.plot([i for i in range(len(curve))], curve, label="i")
    # plt.plot([i for i in range(len(smooth_test_data))], smooth_test_data, label='test')
    # plt.legend()
    # plt.show()
    train_seqs, train_targets = build_dataset(smooth_train_data, seq_len, N)
    test_seqs, test_targets = build_dataset(smooth_test_data, seq_len, N)
    print("Train Dataset:", train_seqs.shape, train_targets.shape, "Test Dataset:", test_seqs.shape, test_targets.shape)

    train_set = SeqDataset(train_seqs, train_targets)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.num_worker, shuffle=False)

    test_set = SeqDataset(test_seqs, test_targets)
    test_loader = DataLoader(test_set, batch_size=args.valid_batch, num_workers=args.num_worker, shuffle=False)

    criterion = nn.MSELoss()
    score_model = MLP(args.top_k, args.top_k)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=args.lr)

    score_model.train()
    min_valid_loss = float('inf')
    min_valid_epoch = 0
    for epoch in range(args.epoch):
        print("Epoch", epoch)
        # Train
        print("Start training")
        sum_loss, batch_num = 0, 0
        self_curve_len = len(smooth_train_data)
        for _, (seq, target) in enumerate(train_loader):
            loss = 0
            start_soh = seq[:, -1, :]
            for i in range(len(start_soh)):
                retrieval_set = build_retrieval_set(curve_funcs, curve_lens, start_soh[i].item(), seq_len)
                x = seq[i].numpy().reshape(-1)
                scores = []
                for retrieval_seq in retrieval_set:
                    alignment = dtw(x, retrieval_seq, keep_internals=True) # DTW算法
                    scores.append(alignment.distance)
                scores = np.array(scores)
                chosen_scores, indices = torch.topk(torch.Tensor(-scores), args.top_k) # 乘 -1 找最小的
                chosen_scores = -chosen_scores
                # print(chosen_scores)
                max_ = torch.max(chosen_scores)
                min_ = torch.min(chosen_scores)
                chosen_scores = (chosen_scores - min_) / (max_ - min_)
                output = score_model(chosen_scores)
                # weight = 1 / (1 + output)
                # print(chosen_scores, output, indices)
                error, diff = custom_loss(output, chosen_scores)
                loss_0 = score_model.loss_weights[0] * error + score_model.loss_weights[1] * diff

                target_cycle = target[i][0].float()
                target_soh = target[i][1]
                # TODO：这里的权重到最后权重的映射不是线性的
                output_clone = output.clone()
                for n in range(len(indices)):
                    curve_id = indices[n]
                    start_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], start_soh[i].data.item()))
                    predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                    output_clone[n] *= (predict_cycle.x.item() - start_cycle.x.item()) * curve_lens[curve_id]
                output = output_clone
                # loss += criterion(output.sum() / self_curve_len, target_cycle / self_curve_len)
                loss_1 = criterion(output.sum(), target_cycle) 
                loss += loss_0 + loss_1

            loss /= args.batch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sum_loss += loss.item()
            batch_num += 1
        print("train_average_loss", sum_loss / batch_num)
        # Eval
        print("Start validating")
        score_model.eval()
        loss = 0
        batch_num = 0
        predict_list, target_list = [], []
        self_curve_len = len(smooth_test_data)
        for step, (seq, target) in enumerate(test_loader):
            if step == 0:
                for i in range(len(seq[0])):
                    predict_list.append(i)
                    target_list.append((i, seq[0][i].item()))
            start_soh = seq[:, -1, :]
            retrieval_set =  build_retrieval_set(curve_funcs, curve_lens, start_soh.item(), seq_len)
            x = seq.numpy().reshape(-1)

            #encoder,contrastic loss
            #########score############
            scores = []
            for retrieval_seq in retrieval_set:
                alignment = dtw(x, retrieval_seq, keep_internals=True) # DTW算法
                scores.append(alignment.distance)
            scores = np.array(scores)


            chosen_scores, indices = torch.topk(torch.Tensor(-scores), args.top_k) # 乘 -1 找最小的
            chosen_scores = -chosen_scores
            max_ = torch.max(chosen_scores)
            min_ = torch.min(chosen_scores)
            chosen_scores = (chosen_scores - min_) / (max_ - min_)
            output = score_model(chosen_scores)
            # print(chosen_scores, output, indices)
            target = target.flatten()
            target_cycle = target[0].float()
            target_soh = target[1]
            #TODO：这里的权重到最后权重的映射不是线性的
            output_clone = output.clone()
            for n in range(len(indices)):
                curve_id = indices[n]
                start_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], start_soh.item()))
                predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                output_clone[n] *= (predict_cycle.x.item() - start_cycle.x.item()) * curve_lens[curve_id]
            if step % 40 == 0:
                print(output_clone.sum(), target_cycle)
            output = output_clone
            # print(output_clone.sum().detach().item() * self_curve_len, target_cycle.detach().item())
            target_list.append((target_cycle.detach().numpy(), target_soh.detach().numpy()))
            predict_list.append(step + seq_len + output_clone.sum().detach().item())
            loss += (output_clone.sum().detach().item() - target_cycle.detach().item())
            batch_num += 1
        print("test_average_loss", loss / batch_num)
        # for i in range(len(target_list)):
        #     print(target_list[i], predict_list[i])
        # assert 0
        if abs(loss / batch_num) < min_valid_loss:
            min_valid_loss = abs(loss / batch_num)
            min_valid_epoch = epoch
            plt.plot([i for i in range(len(smooth_test_data))], smooth_test_data)
            plt.plot([i for i in predict_list], [soh for cycle, soh in target_list])
            save_path = f"./figures/seq_len={seq_len}&N={N}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig('/'.join([save_path, f'epoch{epoch}_{min_valid_loss}.png']))
            plt.close()
    with open(f"./reports/seq_len={seq_len}&N={N}.txt", 'w') as f:
        f.write(f"Summary: min_valid_loss: {min_valid_loss} in epoch: {min_valid_epoch}, relative_min_loss: {min_valid_loss / N * 100}%")
        f.close()

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-seq_len", help="The sequence length", default=100, type=int)
    arg_parser.add_argument("-N", help="Predict the next N cycle's SoH", default=1, type=int)
    arg_parser.add_argument("-batch", help="batch_size", default=16, type=int)
    arg_parser.add_argument("-valid_batch", help="batch_size", default=1, type=int)
    arg_parser.add_argument("-num_worker", help="number of worker", default=0, type=int)
    arg_parser.add_argument("-epoch", help="num of epoch", default=10, type=int)
    arg_parser.add_argument("-lr", help="learning rate", default=1e-3, type=float)
    arg_parser.add_argument("-top_k", help="choose top_k to retrieval", default=3, type=int)

    args = arg_parser.parse_args()

    curves, curve_funcs, curve_lens = data_aug()
    # print(curve_lens)
    # assert 0
    print("Curves number after augmentation:", len(curves))
    
    for seq_len in [50, 75, 100, 125]:
    # seq_len = int(args.seq_len)
        for N in [10, 50, 80, 100]:
    # N = int(args.N)
            run(seq_len, N, curve_lens, curve_funcs, args)

            

if __name__ == "__main__":
    main()