import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt

def interpolate(data, scale_rate=1):

    # 原始数据的长度
    original_length = len(data)    

    # 你希望的新数据的长度（降采样后）
    new_length = int(original_length * scale_rate)

    # 创建原始数据的x坐标
    x = np.linspace(0, 1, original_length)  

    # 创建插值函数
    f = interp1d(x, data.squeeze(1))  

    spline = UnivariateSpline(x, data)

    # 创建新的x坐标
    new_x = np.linspace(0, 1, new_length)

    # 使用插值函数计算新的y坐标（数据）
    new_data = f(new_x) 

    new_spline = spline(new_x)

    # print("Original data:", data)
    # print("New data:", new_data)
    return new_data, new_spline, new_length, f, spline, original_length

def data_aug(path):
    train_data = np.load('/'.join([path, 'soh.npy']))
    scale_rates = [0.8, 0.9, 0.93, 0.96, 0.98, 1.02, 1.05, 1.08, 1.1, 1.2, 1.3]
    curve_list, s_curve_list, curve_func_list, s_curve_func_list, new_length_list, origin_length_list = [], [], [], [], [], []
    for scale_rate in scale_rates:
        curve, smooth_curve, new_length, curve_f, smooth_curve_f, original_length = interpolate(train_data, scale_rate)
        curve_list.append(curve)
        s_curve_list.append(smooth_curve)
        curve_func_list.append(curve_f)
        s_curve_func_list.append(smooth_curve_f)
        new_length_list.append(new_length)
        origin_length_list.append(original_length)
    #     plt.plot([i for i in range(len(curve))], curve)
    #     plt.plot([i for i in range(len(curve))], smooth_curve)
    # plt.show()
    return s_curve_list, s_curve_func_list, new_length_list


if __name__ == "__main__":
    data_aug()
        