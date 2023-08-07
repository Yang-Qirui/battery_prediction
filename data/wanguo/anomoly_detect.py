import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import argparse

SEQ_LEN = 100
DEFAULT_OUTLIER_CURVE = 1

def ts_decompose(data):
    time_index = pd.date_range(start='2022-05-01', periods=len(data))
    time_series = pd.Series(data, index=time_index)
    # 对时间序列进行分解
    result = seasonal_decompose(time_series, model='additive')
    return result

# 定义计算离群值的函数
def calculate_outlier_values(data):
    print(data.shape)
    outlier_values = []
    for i in range(data.shape[0]):
        sequence = data[:i+1]
        distances = []
        for j in range(data.shape[1]):
            if j == i:
                continue
            other_sequence = data[:i+1, j]
            distance, path = fastdtw(sequence, other_sequence, dist=euclidean)
            distances.append(distance)
        outlier_values.append(np.mean(distances))
    return np.array(outlier_values)

def main(contamination, path):
    # 读取.npy文件
    data = np.load(path)

    random_indices = random.sample(range(0, data.shape[1]-1), DEFAULT_OUTLIER_CURVE) 
    random_indices.sort()
    print("ERROR:", random_indices)

    # 生成扰动
    for random_index in random_indices:
        start = int((0.1 + 0.4 * np.random.rand()) * len(data[:, random_index]))
        disturbance_length = len(data[:, random_index]) - start
        disturbance = np.array([(1.00001 + 0.00009 * np.random.rand()) ** i for i in range(disturbance_length)])
        data[start:, random_index] *= disturbance

    decomposed = [seasonal_decompose(data[:, i], model='additive', period=1) for i in range(data.shape[1])]
    trends = [decomp.trend for decomp in decomposed]
    trends = np.array(trends)

    mask = np.isin(range(data.shape[-1]), random_indices)
    normal_trends = trends[~mask]
    abnorm_trends = trends[mask]

    from sklearn.neighbors import KernelDensity
    # train_trends = np.transpose(normal_trends)
    probs_list = []
    for t in range(SEQ_LEN, trends.shape[1]):
        partial_trends = normal_trends[:, t - SEQ_LEN:t]
        # print(normal_trends.shape, abnorm_trends.shape)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kde.fit(partial_trends)
        log_probs = kde.score_samples(trends[:,t - SEQ_LEN:t])
        scaled_log_probs = (log_probs - np.min(log_probs)) / (np.max(log_probs) - np.min(log_probs))
        # threshold = np.percentile(log_probs, int(len(random_indices) / len(trends) * 100))
        # outlier_indices = np.where(log_probs < threshold)[0]
        probs_list.append(scaled_log_probs)

    probs_list = np.array(probs_list).transpose()
    decomposed_probs = [seasonal_decompose(probs_list[i], model='additive', period=1) for i in range(probs_list.shape[0])]
    prob_trends = [decomp.trend for decomp in decomposed_probs]

    clf = IsolationForest(contamination=contamination).fit(probs_list)

    # 预测异常的KDE曲线
    pred = clf.predict(probs_list)

    # 找出被认为是异常的曲线的索引
    outlier_indices = np.where(pred == -1)[0]
    print("PREDICT:", outlier_indices)
    # plt.legend()
    true_pos = set(outlier_indices) & set(random_indices)
    precision = len(list(true_pos)) / len(outlier_indices)
    recall = len(list(true_pos)) / len(random_indices)
    print("precision:", precision, "recall:", recall)
    # plot
    if args.plot:
        for i, prob_trend in enumerate(prob_trends):
            # print(prob_curve)
            # plt.plot([i for i in range(len(prob_curve.trend))], prob_curve.trend, label=f'{i}')
            plt.plot(prob_trend)
        plt.xlabel("cycle")
        plt.ylabel("prob")
        plt.show()

    return precision, recall

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-contamination", default=0.005, type=float)
    arg_parser.add_argument("-path", default="./group0_features.npy", type=str)
    arg_parser.add_argument("-epoch", default=20, type=int)
    arg_parser.add_argument("-train", help="get best_contamination", default=False, type=bool)
    arg_parser.add_argument("-plot", help="get plot", default=False, type=bool)

    args = arg_parser.parse_args()

    if args.train:
        state_dict = {}
        for epoch in range(args.epoch):
            max_precision, max_recall = 0, 0
            best_contamination = 0
            print("Epoch 1")
            for i,  contamination in enumerate([0.07 + 0.01 * j for j in range(10)]):
                # print("Test:", i)
                precision, recall = main(contamination, args.path)
                if precision > max_precision and recall > max_recall:
                    max_precision = precision
                    max_recall = recall
                    best_contamination = contamination

            print("Best:", max_precision, max_recall, best_contamination)
            if best_contamination in  state_dict.keys():
                state_dict[best_contamination] += 1
            else:
                state_dict[best_contamination] = 1
        
        print("Best contamination", best_contamination)
    else:
        main(args.contamination, args.path)