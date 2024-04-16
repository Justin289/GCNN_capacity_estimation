#先验证容量标签是否存在
import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
import random

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
    def __init__(self, all_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/all_car_dict.npz.npy',
                 ind_ood_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/ind_odd_dict2.npz.npy',
                 train=True, fold_num=2, train_size_ratio = 1.0, file_ratio = 1.0):
        self.all_car_dict = np.load(all_car_dict_path, allow_pickle=True).item()
        ind_ood_car_dict = np.load(ind_ood_car_dict_path, allow_pickle=True).item()
        self.ind_car_num_list = ind_ood_car_dict['ind_sorted']
        self.ood_car_num_list = ind_ood_car_dict['ood_sorted']

        car_num_list = self.ind_car_num_list + self.ood_car_num_list  

        # 筛选有容量标签的车辆编号
        capacity_valid_car_numbers = set()
        base_path = '/log/weiyian/' #your data path
        for each_num in tqdm(car_num_list):
            for each_pkl in self.all_car_dict[each_num]:
                absolute_path = os.path.join(base_path, each_pkl)
                train1 = torch.load(absolute_path)
                if train1[1]["capacity"] != 0:
                    capacity_valid_car_numbers.add(each_num)

        # 将有容量标签的车辆编号转换为列表并排序，确保顺序一致
        capacity_valid_car_numbers = sorted(list(capacity_valid_car_numbers))
        print(capacity_valid_car_numbers)

        # 固定随机种子以确保测试集的一致性
        random.seed(42)  # 使用固定的随机种子
        total_data_size = len(capacity_valid_car_numbers)
        test_size = int(total_data_size * 0.2)  # 20%的数据用作测试集

        # 随机选择测试集的车辆编号
        test_car_numbers = random.sample(capacity_valid_car_numbers, test_size)

        # 移除测试集的车辆编号以获得训练集的车辆编号
        train_car_numbers = [num for num in capacity_valid_car_numbers if num not in test_car_numbers]

        if train:
            # 根据train_size调整训练集的大小
            train_size = int(len(train_car_numbers) * train_size_ratio)  # train_size_ratio由外部定义
            # 因为测试集是固定的，所以这里直接从train_car_numbers中选择前N个作为训练集
            train_size = max(1, train_size)
            car_number = train_car_numbers[:train_size]
            print("Training car numbers:", len(car_number), car_number)
        else:
            # 使用之前随机选择的测试集车辆编号
            car_number = test_car_numbers
            print("Testing car numbers:", len(car_number), car_number)


        self.battery_dataset = []

        capacity_valid_car_number = []

        print("Loading data")
        base_path = '/log/weiyian/'
        for each_num in tqdm(car_number):
            for each_pkl in self.all_car_dict[each_num]:
                absolute_path = os.path.join(base_path, each_pkl)
                train1 = torch.load(absolute_path)
                if train1[1]["capacity"] != 0:
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
