# -*- coding: utf-8 -*-
# @Time : 2022/03/30 10:15
# @Author : cuichaoyu
# @Email : cuichaoyu@thinkenergy.net.cn
# @Project : esb
# @File : pre_utils.py

import os
import re
import numpy as np
import pandas as pd
import time
from scipy import interpolate as ip
from sklearn import preprocessing
from collections import OrderedDict


class Normalizer:
    def __init__(self, dfs=None, params=None):
        """
        归一化
        :param dfs: list 包含每个dataframe
        """
        res = []
        if dfs is not None:
            res.extend(dfs)
            self.max_norm = 0
            self.min_norm = 0
            self.std = 0
            self.mean = 0
            self.compute_min_max(res)
        elif params is not None:
            self.max_norm = np.array(params['max_norm'])
            self.min_norm = np.array(params['min_norm'])
            self.std = np.array(params['std'])
            self.mean = np.array(params['mean'])
        else:
            raise Exception("df list not specified")
       

    def compute_min_max(self, res):


        column_max_all = np.max(res, axis=1)
        column_min_all = np.min(res, axis=1)
        column_std_all = np.std(res, axis=1)
        column_mean_all = np.mean(res, axis=1)
        self.max_norm = np.max(column_max_all, axis=0) + 0.00001
        self.min_norm = np.min(column_min_all, axis=0) - 0.00001
        self.std = np.mean(column_std_all, axis=0)
        self.mean = np.mean(column_mean_all, axis=0)

    def std_norm_df(self, df):
        return (df - self.mean) / np.maximum(1e-4, self.std)

    def norm_func(self, df, norm_name):
        """
        归一化函数
        :param df: dataframe m * n
        :param norm_name: 归一化子类的前缀名
        :return: 调用子类的归一化函数的结果
        """
        return eval(norm_name.capitalize() + 'Normalizer.norm')(self, df)



class EDNormalizer():
    """
    标签归一化，提供两种归一化方式：极小极大归一化与标准差归一化
    """

    def __init__(self):
        """
        初始化
        :param label_path: 存放label文件的路径
        """
        # label_lst = sorted(os.listdir(label_path))
        # if os.path.join(label_path, label_lst[0]).endswith('.csv'):
        #     label_df = pd.read_csv(os.path.join(label_path, label_lst[0]), index_col='file_name')
        # elif os.path.join(label_path, label_lst[0]).endswith('.feather'):
        #     label_df = pd.read_feather(os.path.join(label_path, label_lst[0]), )
        # self.label_df = label_df

    def exp_minmaxscaler(self, min_num=None, max_num=None):
        """
        极小极大归一化，该过程可以指定区间进行，若不指定区间则为整体数据域内进行
        :param min_num: 归一化区间的最小值
        :param max_num: 归一化区间的最大值
        :return: 归一化函数
        """
        # cell_list = [list(self.label_df)[i] for i in range(len(list(self.label_df))) if 'relative_capacity_' in list(self.label_df)[i]]
        # label_df2 = self.label_df[[i for i in cell_list]]
        # if not min_num:
        #     min_num = label_df2.min().min()
        # if not max_num:
        #     max_num = label_df2.max().max()
        temp_label = [[max_num], [min_num]]
        label_normalizer = preprocessing.MinMaxScaler()
        label_normalizer.fit(temp_label)

        return label_normalizer
