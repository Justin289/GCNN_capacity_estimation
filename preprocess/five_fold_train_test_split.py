import numpy as np
import os
import pickle
from glob import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm

data_dir = "/home/weiyian"

#基于battery_dataset生成all_car_dict
#battert_dataset1
data_path = data_dir+'/battery/battery_dataset1/data'
data_pkl_files = glob(data_path+'/*.pkl')

ind_pkl_files = []
ood_pkl_files = []
car_num_list = []

ood_car_num_list1 = set()
ind_car_num_list1 = set()

all_car_dict = {}

for each_path in tqdm(data_pkl_files):
#     print(each_path)
    this_pkl_file = torch.load(each_path)
    this_car_number = this_pkl_file[1]['car']
    if this_pkl_file[1]['label'] == '00':
        ind_pkl_files.append(each_path)
        ind_car_num_list1.add(this_car_number)
    else:
        ood_pkl_files.append(each_path)
        ood_car_num_list1.add(this_car_number)
    car_num_list.append(this_pkl_file[1]['car'])
    if this_car_number not in all_car_dict:
        all_car_dict[this_car_number] = []
        all_car_dict[this_car_number].append(each_path)
    else:
        all_car_dict[this_car_number].append(each_path)

# shuffle
random.seed(0)
ind_sorted = sorted(ind_car_num_list1)
random.shuffle(ind_sorted)
print(ind_sorted)
ood_sorted = sorted(ood_car_num_list1)
random.shuffle(ood_sorted)
print(ood_sorted)        
ind_odd_dict = {}
ind_odd_dict["ind_sorted"],ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs('/log/weiyian/finetuning/preprocess/five_fold_utils', exist_ok=True)
np.save('/log/weiyian/finetuning/preprocess/five_fold_utils/ind_odd_dict1.npz', ind_odd_dict)

#battery_dataset2
data_path = data_dir+'/battery/battery_dataset2/data'

data_pkl_files = glob(data_path+'/*.pkl')
ind_pkl_files = []
ood_pkl_files = []
car_num_list = []

ood_car_num_list2 = set()
ind_car_num_list2 = set()

for each_path in tqdm(data_pkl_files):
#     print(each_path)
    this_pkl_file = torch.load(each_path)
    this_car_number = this_pkl_file[1]['car']
    if this_pkl_file[1]['label'] == '00':
        ind_pkl_files.append(each_path)
        ind_car_num_list2.add(this_car_number)
    else:
        ood_pkl_files.append(each_path)
        ood_car_num_list2.add(this_car_number)
    car_num_list.append(this_pkl_file[1]['car'])
    if this_car_number not in all_car_dict:
        all_car_dict[this_car_number] = []
        all_car_dict[this_car_number].append(each_path)
    else:
        all_car_dict[this_car_number].append(each_path)

print(ind_car_num_list2, len(ind_car_num_list2))
print(ood_car_num_list2, len(ood_car_num_list2))

# shuffle
random.seed(0)
ind_sorted = sorted(ind_car_num_list2)
random.shuffle(ind_sorted)
print(ind_sorted)
ood_sorted = sorted(ood_car_num_list2)
random.shuffle(ood_sorted)
print(ood_sorted)        
ind_odd_dict = {}
ind_odd_dict["ind_sorted"],ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs('/log/weiyian/finetuning/preprocess/five_fold_utils', exist_ok=True)
np.save('/log/weiyian/finetuning/preprocess/five_fold_utils/ind_odd_dict2.npz', ind_odd_dict)

#battery_dataset3
data_path = data_dir+'/battery/battery_dataset3/data'

data_pkl_files = glob(data_path+'/*.pkl')

ind_pkl_files = []
ood_pkl_files = []
car_num_list = []

ood_car_num_list3 = set()
ind_car_num_list3 = set()

for each_path in tqdm(data_pkl_files):
#     print(each_path)
    this_pkl_file = torch.load(each_path)
    this_car_number = this_pkl_file[1]['car']
    if this_pkl_file[1]['label'] == '00':
        ind_pkl_files.append(each_path)
        ind_car_num_list3.add(this_car_number)
    else:
        ood_pkl_files.append(each_path)
        ood_car_num_list3.add(this_car_number)
    car_num_list.append(this_pkl_file[1]['car'])
    if this_car_number not in all_car_dict:
        all_car_dict[this_car_number] = []
        all_car_dict[this_car_number].append(each_path)
    else:
        all_car_dict[this_car_number].append(each_path)


print(ind_car_num_list3, len(ind_car_num_list3))
print(ood_car_num_list3, len(ood_car_num_list3))

# shuffle
random.seed(0)
ind_sorted = sorted(ind_car_num_list3)
random.shuffle(ind_sorted)
print(ind_sorted)
ood_sorted = sorted(ood_car_num_list3)
random.shuffle(ood_sorted)
print(ood_sorted)
ind_odd_dict = {}
ind_odd_dict["ind_sorted"],ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs('/log/weiyian/finetuning/preprocess/five_fold_utils', exist_ok=True)
np.save('/log/weiyian/finetuning/preprocess/five_fold_utils/ind_odd_dict3.npz', ind_odd_dict)

#all cars random shuffle
# shuffle
random.seed(0)
ind_sorted = sorted(list(ind_car_num_list1)+list(ind_car_num_list2)+list(ind_car_num_list3))
random.shuffle(ind_sorted)
print(ind_sorted)
ood_sorted = sorted(list(ood_car_num_list1)+list(ood_car_num_list2)+list(ood_car_num_list3))
random.shuffle(ood_sorted)
print(ood_sorted)
# print(ind_car_num_list, len(ind_car_num_list))
# print(ood_car_num_list, len(ood_car_num_list))

# save all the three brands path information
np.save('/log/weiyian/finetuning/preprocess/five_fold_utils/all_car_dict.npz', all_car_dict)

ind_odd_dict = {}
ind_odd_dict["ind_sorted"],ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs('/log/weiyian/finetuning/preprocess/five_fold_utils', exist_ok=True)
np.save('/log/weiyian/finetuning/preprocess/five_fold_utils/ind_odd_dict.npz', ind_odd_dict)