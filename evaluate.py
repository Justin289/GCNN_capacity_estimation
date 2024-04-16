import os
import time
import sys
import json
import argparse
import torch
import random
import warnings

import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.getcwd())
# sys.path.insert(0, os.path.dirname(os.getcwd()))
from model.arch import gcn
from preprocess.dataset import SlidingWindowBattery, PreprocessNormalizer
from preprocess.capacity_dataset2 import CapacityDataset
from preprocess.pre_utils import Normalizer, LabelNormalizer, PredictResult, EDNormalizer, EDPredictResult
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import random
import warnings
from loguru import logger
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_percentage_error


class Evaluate:
    """
    验证模块
    """

    def __init__(self, args):
        """
        初始化
        :param args: 包含项目信息 如数据、各种配置参数等
        """
        self.mode="test"
        self.args = args
        # self.eval_data_path="/data/nfsdata/database/FENGCHAO/batch1/volt88_temp32/1ep_s10_d50_t60/all_data/csv/"
        #self.data_list = pd.read_csv(self.args.eval_label_path, engine="python").dropna(axis=0,subset=[f"{self.mode}_file"])[f"{self.mode}_file"].values
        #logger.info(f"test的长度为：{len(self.data_list)}")
        self.normalizer = None


    def main(self):

        model_torch = os.path.join(self.args.current_model_path, "model.torch")
        data_eval = CapacityDataset(
            all_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/all_car_dict.npz.npy',
            ind_ood_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/ind_odd_dict3.npz.npy',
            train=False,
            #train_size_ratio=self.args.train_size_ratio, 
            #file_ratio = self.args.file_ratio,
            fold_num = 0#self.args.fold_num
        )
        params = dict(
            device=self.args.device,
            embed_dim=self.args.embed_dim,
            kernel_sizes=self.args.kernel_sizes,
            drop_out=self.args.drop_out,
            last_kernel_num=self.args.last_kernel_num,
        )


        model = to_var(gcn.CNN_Gate_Aspect_Text(**params), self.args.device).float()
        model.load_state_dict(torch.load(model_torch))

        norm_json = json.load(open(os.path.join(self.args.current_model_path, "norm.json"), 'r'))
        self.label_normalizer = EDNormalizer().exp_minmaxscaler(min_num=0, max_num=100)
        self.normalizer = Normalizer(dfs=None, params=norm_json)
        eval_pre = PreprocessNormalizer(data_eval, norm_name=self.args.norm, normalizer_fn=self.normalizer.norm_func)
        eval_loader = DataLoader(dataset=eval_pre, batch_size=self.args.batch_size, shuffle=False,
                                  num_workers=self.args.jobs, drop_last=False, )
        model.eval()
        logger.info("start evaluate")
        result_list = self.running_program(model,eval_loader)
        mape=mean_absolute_percentage_error(result_list[3], result_list[2])
        logger.info(f"mape:{mape}")
        rmse=np.sqrt(mean_squared_error(result_list[2], result_list[3]))
        logger.info(f"rmse:{rmse}")
        mae=mean_absolute_error(result_list[2], result_list[3])
        logger.info(f"mae:{mae}")
        self.save_predict_result(result_list)


    def running_program(self, model,data_loader):

        result_list = [[],[],[],[]]

        for idx, (input_data, metadata) in enumerate(tqdm(data_loader)):
            outputs = model(to_var(input_data, self.args.device))
            restored_outputs = self.label_normalizer.inverse_transform(outputs.detach().cpu().numpy())

            files = np.array(metadata['car']).squeeze()
            result_list[0].extend(files)

            mileage = np.array(metadata['mileage']).squeeze()
            result_list[1].extend(mileage)
            pre_cap = restored_outputs.squeeze()
            result_list[2].extend(pre_cap)
            soh = np.array(metadata['capacity']).squeeze()
            result_list[3].extend(soh)

        return result_list

    def save_predict_result(self, result_list):

        result_df = pd.DataFrame({
            'file': result_list[0],
            'mileage': result_list[1],
            'soh_pred': result_list[2],
            "soh": result_list[3]
        })
        # 定义保存路径
        save_path = '/home/weiyian/finetuning/result'

        # 保存文件
        result_df.to_csv(f"{save_path}/{self.mode}_data3_beforetuning_result.csv", index=False)



def to_var(x, device='cpu'):
    """
    如果有gpu将x放入cuda中
    :param x: data or model
    :param device cpu / gpu
    :return: 放入cuda后的x
    """
    if device == 'cuda':
        x = x.cuda()
    return x



if __name__ == '__main__':


    parser = argparse.ArgumentParser()


    parser.add_argument('--model_params', type=str,
                        default="/log/weiyian/log/gate_cnn/data1/2024-03-31-00-51-02/model/model_params.json")


    
    args = parser.parse_args()
    with open(args.model_params, 'r') as file:
        args.__dict__.update(json.load(file)["args"])

    Evaluate(args).main()




