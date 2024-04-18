import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm

#print("当前工作目录:", os.getcwd())
# 如果需要，改变工作目录
# os.chdir('预期的路径')

class CapacityDataset:
    '''
    If you want to use another vendor, just switch paths
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict1.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict2.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict3.npz.npy'
    '''
    def __init__(self, all_car_dict_path='finetuning/preprocess/five_fold_utils/all_car_dict.npz.npy',
                 ind_ood_car_dict_path='finetuning/preprocess/five_fold_utils/ind_odd_dict2.npz.npy',
                 train=True, fold_num=2, train_size_ratio = 0.8, file_ratio = 1.0):
        self.all_car_dict = np.load(all_car_dict_path, allow_pickle=True).item()
        ind_ood_car_dict = np.load(ind_ood_car_dict_path, allow_pickle=True).item()
        self.ind_car_num_list = ind_ood_car_dict['ind_sorted']
        self.ood_car_num_list = ind_ood_car_dict['ood_sorted']

        # 确定训练集和测试集范围
        ind_train_start = int(fold_num * len(self.ind_car_num_list) / 5)
        ind_train_end = int((fold_num + 1) * len(self.ind_car_num_list) / 5)
        ood_train_start = int(fold_num * len(self.ood_car_num_list) / 5)
        ood_train_end = int((fold_num + 1) * len(self.ood_car_num_list) / 5)
        
        if train:
            # Combine ind and ood car numbers except the fold reserved for testing
            car_number = self.ind_car_num_list[:ind_train_start] + self.ind_car_num_list[ind_train_end:] + \
                         self.ood_car_num_list[:ood_train_start] + self.ood_car_num_list[ood_train_end:]
            # 如果需要验证train_size对模型的影响，将下两行加入代码中即可
            # 1. 根据train_size_ratio随机选取训练集
            #random_car = random.sample(range(len(car_number)), int((len(car_number) * train_size_ratio)) / 0.8)
            #car_number = [car_number[i] for i in random_car]
            # 2. 根据train_size从后往前取
            #actual_train_size = int((len(car_number) * (1-train_size_ratio)) / 0.8)

            #car_number = combined_car_number[actual_train_size:]
        else:  # for testing
            # Use only the fold reserved for testing
            car_number = self.ind_car_num_list[ind_train_start:ind_train_end] + \
                         self.ood_car_num_list[ood_train_start:ood_train_end]

        self.battery_dataset = []

        capacity_valid_car_number = []

        print("Loading data")
        base_path = 'finetuning/data'
        for each_num in tqdm(car_number):
            for each_pkl in self.all_car_dict[each_num]:
                absolute_path = os.path.join(base_path, each_pkl)
                train1 = torch.load(absolute_path)
                if train1[1]["capacity"] != 0:
                    adjusted_file_length = int(len(train1) * file_ratio)
                    # 1. file_ratio取值方式：从后往前取file_ratio%
                    #start_index = len(train1) - adjusted_file_length
                    #adjusted_train1 = train1[start_index:]
                    # 2. file_ratio取值方式：从前往后取file_ratio%
                    adjusted_train1 = train1[:adjusted_file_length] 
                    # 2. file_ratio取值方式：从中间取file_ratio%
                    #start_index = len(train1) // 2 - adjusted_file_length // 2
                    #end_index = start_index + adjusted_file_length
                    #adjusted_train1 = train1[start_index:end_index]
                    self.battery_dataset.append(train1)
                    capacity_valid_car_number.append(train1[1]["car"])
        print("Sorted capacity_valid_car_number",
              len(set(capacity_valid_car_number)), set(capacity_valid_car_number),
              "Capacity data point",
              len(capacity_valid_car_number))

    def __len__(self):
        return len(self.battery_dataset)

    def __getitem__(self, idx):
        file = self.battery_dataset[idx]
        return file

