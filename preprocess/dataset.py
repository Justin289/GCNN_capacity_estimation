import os
import numpy as np
import pandas as pd
import multiprocessing
import random
from scipy import interpolate as ip
from collections import OrderedDict
from decimal import Decimal
import sys



class PreprocessNormalizer:
    """
    数据归一化类
    """

    def __init__(self, dataset, norm_name=None, normalizer_fn=None):
        """
        初始化
        :param dataset: SlidingWindowBattery
        :param norm_name: 用哪种归一化 如 ev 表示 EvNormalizer
        :param normalizer_fn: 归一化函数
        """
        self.dataset = dataset
        self.norm_name = norm_name
        self.normalizer_fn = normalizer_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        df, label = self.dataset[idx][0], self.dataset[idx][1]
        if self.normalizer_fn is not None:
            df = self.normalizer_fn(df, self.norm_name)
        return df, label

    def get_column(self):
        df = self.dataset[1][0]
        return list(df.columns)







