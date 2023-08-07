import glob
import pandas as pd
import numpy as np
import re

def get_id(list, pattern):
    first_match_index = None
    last_match_index = None
    for i, string in enumerate(list):
        if len(re.findall(pattern, string)) > 0:
            if first_match_index is None:
                first_match_index = i
            last_match_index = i

    assert first_match_index is not None and last_match_index is not None
    return first_match_index, last_match_index

def s_k(x, avg, std):
    return (x - avg) ** 3 / (std ** 3)

def k_k(x, avg, std):
    return (x - avg) ** 4 / (std ** 4)

def get_items_feature(pattern, features, data):
    s, e = get_id(features, pattern)
    pos_temp = data[s: e]
    avg_pos_temp = pos_temp.mean()
    std_pos_temp = pos_temp.std()
    sk_pos_temp = pos_temp.apply(s_k, args=(avg_pos_temp, std_pos_temp)).mean()
    kk_pos_temp = pos_temp.apply(k_k, args=(avg_pos_temp, std_pos_temp)).mean()
    return [avg_pos_temp, std_pos_temp, sk_pos_temp, kk_pos_temp]

def main():

    cycles = 0
    cycle_list = []

    folder_path = './south'  # 替换为实际的文件夹路径

    # 使用 glob 模块匹配所有 .xlsx 文件
    xlsx_files = sorted(glob.glob(folder_path + '/*.xlsx'))

    # 使用 Pandas 打开每个 .xlsx 文件

    equip_features = [[], [], []]
    equip_targets = [[], [], []]


    for file in xlsx_files:
        df = pd.read_excel(file)
        print(f"文件名: {file}")

        # 有若干个电池组，分别算出哪些数据属于哪个电池组
        all_equipments = df['设备'].tolist()
        all_equipment_name = list(set(all_equipments))
        equip_index_range = []
        for equip in all_equipment_name:
            equip_index_range.append(len(all_equipments) - list(reversed(all_equipments)).index(equip)) # 不 -1，因为后面访问时是左闭右开
        equip_index_range.sort()
        print("segments:", equip_index_range)
       
        all_feature_name = df['单位'].tolist()
         
        for column, column_data in df.items():
            start_index = 0
            if re.findall(r'\d{4}-\d{1,2}-\d{1,2}', column): # 该列是某一天的数据
                for (i, end_index) in enumerate(equip_index_range):
                    features = []
                    targets = []

                    sub_feature_name = all_feature_name[start_index: end_index]
                    sub_col_data = column_data[start_index: end_index].reset_index(drop=True)

                    targets.append(sub_col_data[sub_feature_name.index('系统SOH')])
                    # features.append(sub_col_data[sub_feature_name.index('系统SOH')])

                    # features.append(sub_col_data[sub_feature_name.index('系统总电压_V')])
                    # features.append(sub_col_data[sub_feature_name.index('系统平均电压')])
                    # features.append(sub_col_data[sub_feature_name.index('系统平均温度')])

                    # 1簇feature
                    # features.append(sub_col_data[sub_feature_name.index('1_簇并机总电压')])
                    for j in range(1, 160):
                        features.append(sub_col_data[sub_feature_name.index(f'1_簇单体电压{j}')])
                    # features += get_items_feature(r'1_模组\d{1,2}正极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'1_模组\d{1,2}负极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'1_簇单体电压\d{1,2}', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'1_簇单体温度\d{1,2}', sub_feature_name, sub_col_data)

                    # # 2簇feature
                    # features.append(sub_col_data[sub_feature_name.index('2_簇并机总电压')])
                    # features.append(sub_col_data[sub_feature_name.index('2_簇电池总电压')])
                    # features += get_items_feature(r'2_模组\d{1,2}正极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'2_模组\d{1,2}负极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'2_簇单体电压\d{1,2}', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'2_簇单体温度\d{1,2}', sub_feature_name, sub_col_data)

                    # if i != 0:
                    #     # 3簇feature
                    #     features.append(sub_col_data[sub_feature_name.index('3_簇并机总电压')])
                    #     features.append(sub_col_data[sub_feature_name.index('3_簇电池总电压')])
                    #     features += get_items_feature(r'3_模组\d{1,2}正极柱温度', sub_feature_name, sub_col_data)
                    #     features += get_items_feature(r'3_模组\d{1,2}负极柱温度', sub_feature_name, sub_col_data)
                    #     features += get_items_feature(r'3_簇单体电压\d{1,2}', sub_feature_name, sub_col_data)
                    #     features += get_items_feature(r'3_簇单体温度\d{1,2}', sub_feature_name, sub_col_data)

                    equip_features[i].append(features)
                    equip_targets[i].append(targets)
                    
                    start_index = end_index

                cycles += 1

    for (i, features) in enumerate(equip_features):
        arr = np.array(features)
        print(arr.shape)
        np.save(f'group{i}_features.npy', arr)

    for (i, targets) in enumerate(equip_targets):
        targets = np.array(targets)
        targets = targets.flatten()
        targets = targets[targets != 0]
        arr = targets.reshape(-1, 1)
        print(arr.shape)
        np.save(f'group{i}_targets.npy', arr)
    

if __name__ == "__main__":
    main()