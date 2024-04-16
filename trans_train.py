import os
import time
import sys
import json
import argparse
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from lstmmodel import LSTMNet, MLP
#import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, os.getcwd())
print(os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.dirname(os.getcwd()))
from model.arch import gcn
from preprocess.dataset import PreprocessNormalizer
from preprocess.capacity_dataset import CapacityDataset
from preprocess.pre_utils import Normalizer, PredictResult, EDNormalizer
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import random
import optuna
from loguru import logger


# seed=0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class Train:
    """
    训练模块
    """

    def __init__(self, args):
        """
        初始化
        :param args: 包含项目信息 如数据、各种配置参数等

        """
        self.args = args

        self.current_epoch = 1
        self.step = 1
        self.train_step = 1
        self.test_step = 1
        self.loss_dict = OrderedDict()
        temp_tensor = torch.randn(self.args.batch_size, 1)
        for t_i in range(self.args.batch_size):
            temp_tensor[t_i] = random.randint(2, 5)

        self.random_label = temp_tensor
        # self.args.save_model_path = os.path.join('E:/', 'current_path', self.args.save_model_path)
        self.args.save_model_path = f"/log/{os.getlogin()}/log/gate_cnn/{self.args.data_name}/" #replace to your model save path
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        current_path = os.path.join(self.args.save_model_path, time_now)
        logger.info(f"model save path:{current_path}")
        self.mkdir(current_path)
        self.current_path = current_path
        current_model_path = os.path.join(current_path, "model")
        loss_picture_path = os.path.join(current_path, "loss")
        current_result_path = os.path.join(current_path, "result")
        current_tb_path = os.path.join(current_path, "tb")
        self.mkdir(current_model_path)
        self.mkdir(loss_picture_path)
        self.mkdir(current_result_path)
        self.mkdir(current_tb_path)
        self.args.current_path = current_path
        self.args.current_model_path = current_model_path
        self.args.loss_picture_path = loss_picture_path
        self.args.current_result_path = current_result_path
        self.args.current_tb_path = current_tb_path
        self.writer = SummaryWriter(current_tb_path)
        self.normalizer = None
        self.model = None
        #在跑不同训练集比例和文件比例时：
        #self.args.train_size_ratio = args.train_size_ratio
        #self.args.file_ratio = args.file_ratio
        self.args.fold_num = args.fold_num
        self.train_file = {}
        self.test_file = {}
        self.valid_file = {}
        
    @staticmethod
    def mkdir(path):
        """
        创建目录
        :param path: 要创建的路径
        """
        if os.path.exists(path):
            print('%s is exist' % path)
        else:
            os.makedirs(path)

    def main(self, optuna_params={}):
        """
        训练主程序
        """

        start_time = time.time()

        #self.split_train_test_vin_by_mileage()
        if self.args.optuna:
            self.args.learning_rate = optuna_params['learning_rate']
            self.args.batch_size = optuna_params['batch_size']
            self.args.cosine_factor = optuna_params['cosine_factor']
            self.args.drop_out = optuna_params['drop_out']
            # self.args.last_kernel_num = optuna_params['last_kernel_num']

        params = dict(
            device=self.args.device,
            embed_dim=self.args.embed_dim,
            kernel_sizes=self.args.kernel_sizes,
            drop_out=self.args.drop_out,
            last_kernel_num=self.args.last_kernel_num,
        )
        data_train = CapacityDataset(
            all_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/all_car_dict.npz.npy',
            ind_ood_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/ind_odd_dict2.npz.npy',
            train=True,
            train_size_ratio = self.args.train_size_ratio, 
            file_ratio = self.args.file_ratio,
            fold_num = 0 #self.args.fold_num
        )

        data_test = CapacityDataset(
            all_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/all_car_dict.npz.npy',
            ind_ood_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/ind_odd_dict2.npz.npy',
            train=False,
            train_size_ratio=self.args.train_size_ratio, 
            file_ratio = self.args.file_ratio,
            fold_num = 0 #self.args.fold_num
        )
        #norm_json = json.load(open(os.path.join("/log/weiyian/log/gate_cnn/data1/2024-03-31-10-55-45/model/", "norm.json"), 'r'))
        self.label_normalizer = EDNormalizer().exp_minmaxscaler(min_num=0, max_num=100)
        self.normalizer = Normalizer(dfs=[data_train[i][0] for i in range(len(data_train))])
        train_pre = PreprocessNormalizer(data_train, norm_name=self.args.norm, normalizer_fn=self.normalizer.norm_func)
        test_pre = PreprocessNormalizer(data_test, norm_name=self.args.norm, normalizer_fn=self.normalizer.norm_func)
        print(f'Dataset complete in {time.time() - start_time}s')

        train_loader = DataLoader(dataset=train_pre, batch_size=self.args.batch_size, shuffle=True,
                                      num_workers=0, drop_last=True, )
        test_loader = DataLoader(dataset=test_pre, batch_size=self.args.batch_size, shuffle=True,
                                     num_workers=0, drop_last=True, )

        # with open('./trainloader_save.pkl','rb') as f:
        #     train_loader=dill.load(f)
        # with open('./testloader_save.pkl','rb') as f:
        #     test_loader=dill.load(f)

        self.model = to_var(gcn.CNN_Gate_Aspect_Text(**params), self.args.device).float()
        # logger.info(self.model)
        if self.args.trans_train:
            self.model.load_state_dict(torch.load(self.args.model_path))
            # exit(3333)

        self.writer.add_graph(self.model,
                              to_var(torch.zeros(self.args.batch_size, data_test[0][0].shape[1]),
                                     self.args.device))
        #print("model", self.model)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs,
                                      eta_min=self.args.cosine_factor * self.args.learning_rate)
        criterion = nn.MSELoss()

        train_best_running_loss = float('inf')
        test_best_running_loss = float('inf')
        test_best_rmse = float('inf')
        test_best_mape = float('inf')
        test_best_mae = float('inf')
        epoch_stats = []
        while self.current_epoch <= self.args.epochs:
            # print(f'Epoch{self.current_epoch}:')

            train_running_loss, train_rmse, train_mape_error, train_mae, train_epoch_mape_values, train_epoch_rmse_values = self.running_program(self.current_epoch,
                                                                                                                                 self.args.epochs, mode='train',
                                                                                                                                 data_loader=train_loader,
                                                                                                                                 criterion=criterion,
                                                                                                                                 optimizer=optimizer,
                                                                                                                                 scheduler=scheduler, )

            test_running_loss, test_rmse, test_mape_error, test_mae, test_epoch_mape_values, test_epoch_rmse_values = self.running_program(self.current_epoch, self.args.epochs,
                                                                                                                           mode='test',
                                                                                                                           data_loader=test_loader,
                                                                                                                           criterion=criterion)
            epoch_train_loss = train_running_loss
            epoch_test_loss = test_running_loss
            train_mape_max = max(train_epoch_mape_values)
            train_mape_min = min(train_epoch_mape_values)
            train_mape_avg = sum(train_epoch_mape_values) / len(train_epoch_mape_values)

            train_rmse_max = max(train_epoch_rmse_values)
            train_rmse_min = min(train_epoch_rmse_values)
            train_rmse_avg = sum(train_epoch_rmse_values) / len(train_epoch_rmse_values)
            test_mape_max = max(test_epoch_mape_values)
            test_mape_min = min(test_epoch_mape_values)
            test_mape_avg = sum(test_epoch_mape_values) / len(test_epoch_mape_values)

            test_rmse_max = max(test_epoch_rmse_values)
            test_rmse_min = min(test_epoch_rmse_values)
            test_rmse_avg = sum(test_epoch_rmse_values) / len(test_epoch_rmse_values)
            epoch_stats.append({
                "epoch": self.current_epoch,
                "fold": self.args.fold_num,
                "train_mape_max": train_mape_max,
                "train_mape_min": train_mape_min,
                "train_mape_avg": train_mape_avg,
                "train_rmse_max": train_rmse_max,
                "train_rmse_min": train_rmse_min,
                "train_rmse_avg": train_rmse_avg,
                # 对于测试数据，也添加相应的统计数据
                "test_mape_max": test_mape_max,  # 假设这些值也被计算和存储
                "test_mape_min": test_mape_min,
                "test_mape_avg": test_mape_avg,
                "test_rmse_max": test_rmse_max,
                "test_rmse_min": test_rmse_min,
                "test_rmse_avg": test_rmse_avg,
            })
            self.loss_dict[f"epoch{self.current_epoch}"] = {'epoch_train_loss': epoch_train_loss,
                                                            "epoch_test_loss": epoch_test_loss,
                                                            #############
                                                            # "train_error_rate": train_predict_result.error_rate,
                                                            # "test_error_rate": test_predict_result.error_rate,
                                                            #############
                                                            "train_rmse": train_rmse,
                                                            "test_rmse": test_rmse,
                                                            "train_error": train_mape_error,
                                                            "test_error": test_mape_error,
                                                            "train_mae": train_mae,
                                                            "test_mae": test_mae,
                                                            }
            if test_mape_error <= test_best_mape:
                # if not self.args.optuna:
                self.model_result_save(self.model)
                test_best_mape = test_mape_error,
                test_best_rmse = test_rmse,
                test_best_mae = test_mae
            self.current_epoch += 1
        self.loss_visual()
        print(f"best_mape:{test_best_mape}, best_rmse:{test_best_rmse}, best_mae:{test_best_mae}")
        stats_df = pd.DataFrame(epoch_stats)
        file_path = '/home/weiyian/finetuning/modelresult/dataset2_finetuning_epoch_stats.csv'  # 定义CSV文件路径
        if not os.path.exists(file_path):
            stats_df.to_csv(file_path, index=False)
            print('Statistics have been successfully written to the file with creating a new one.')
        else:
            stats_df.to_csv(file_path, mode="a", header=False, index=False)
            print('Statistics have been successfully appended to the existing file.')
        results = {
            #"train_size_ratio":self.args.train_size_ratio,
            #"file_ratio":self.args.file_ratio,
            "fold_num":self.args.fold_num,
            "best_mape":test_best_mape,
            "best_rmse":test_best_rmse,
            "best_mae":test_best_mae
            }
        results_df = pd.DataFrame([results])
        #根据需要改变保存文件的路径及名称
        file_path = '/home/weiyian/finetuning/modelresult/dataset2_afterfinetuning_trainsize_back123.csv'
        if not os.path.exists(file_path):
            results_df.to_csv(file_path, mode="a", index=False)
            print('success write in with creating')
        else:
            results_df.to_csv(file_path, mode="a", index=False, header=False)
            print('success write in')
        return self.args, test_best_mape, test_best_rmse, test_best_mae

    def running_program(self, epoch, all_epoch, mode, data_loader, criterion, optimizer=None, scheduler=None):
        """
        训练/验证程序
        :param mode: 程序模式，包括train以及test等
        :param data_loader: train或test对应的data_loader
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param scheduler: 学习率衰减策略
        :return: 当前loss误差，预测结果
        """
        # predict_result = EDPredictResult(cluster_nums=self.args.cluster_nums, sigvolt_nums=self.args.sigvolt_nums)
        running_loss, running_mape, iteration, data_len, running_rmse, running_loss_epoch, running_rmse_epoch, running_mape_epoch, running_mae, running_mae_epoch = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        epoch_mape_values = []
        epoch_rmse_values = []
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        pbar = tqdm(data_loader)
        pbar.set_description(f'Epoch:{epoch:03d}/{all_epoch:03d}')
        for idx, (input_data, metadata) in enumerate(pbar):
            input_data = input_data.type(torch.float32)
            outputs = self.model(to_var(input_data, self.args.device))
            origin_labels = metadata['capacity'].reshape(-1, 1)
            labels = self.label_normalizer.transform(origin_labels)
            labels = torch.tensor(labels).to(torch.float32)
            labels = to_var(labels, self.args.device)

            restored_outputs = self.label_normalizer.inverse_transform(outputs.detach().cpu().numpy())
            # predict_result.save_result(output_metadata=metadata, restored_outputs=restored_outputs, )

            loss = criterion(outputs, labels)

            # 更新参数
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            # 函数中会做inverse_transform
            mape = self.calculate_error(restored_outputs, labels)
            # lens = len(labels)
            # data_len += lens
            running_mape += mape
            running_mape_epoch = running_mape / (1 + iteration)

            rmse = np.sqrt(mean_squared_error(outputs.detach().cpu(), labels.cpu()))
            running_rmse += rmse
            running_rmse_epoch = running_rmse / (1 + iteration)

            mae = mean_absolute_error(labels.cpu(), outputs.detach().cpu())
            running_mae += mae
            running_mae_epoch = running_mae / (1 + iteration)

            running_loss += loss.item()
            running_loss_epoch = running_loss / (1 + iteration)
            epoch_mape_values.append(mape)
            epoch_rmse_values.append(rmse)
            loss_info = {'running_loss': running_loss_epoch,
                         'rmse': running_rmse_epoch,
                         'mape': running_mape_epoch,
                         'mae': running_mae_epoch}
            self.tensorboard_loss(loss_info, mode)
            pbar.set_postfix({
                'loss': running_loss_epoch,
                'rmse': running_rmse_epoch,
                'mape': running_mape_epoch,
                'mae': running_mae_epoch})
            self.step += 1
            if mode == 'train':
                self.train_step += 1
            else:
                self.test_step += 1
            iteration += 1

        # predict_result.calculate()

        return running_loss_epoch, running_rmse_epoch, running_mape_epoch, running_mae_epoch, epoch_mape_values, epoch_rmse_values



    def model_result_save(self, model):
        """
        保存模型参数
        :param model: 保存的模型
        """
        model_params = {'train_time_start': self.current_path,
                        'train_time_end': time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                        'args': vars(self.args),
                        'loss': self.loss_dict
                        }

        json.dump(model_params, open(os.path.join(self.args.current_model_path, 'model_params.json'), 'w'),
                  indent=4)

        torch.save(model.state_dict(), os.path.join(self.args.current_model_path, "model.torch"))
        json.dump({k: list(v) for k, v in self.normalizer.__dict__.items()},
                  open(os.path.join(self.args.current_model_path, "norm.json"), 'w'), indent=4)

    def calculate_error(self, output, label):
        label = label.cpu().numpy().reshape((len(label), 1))
        restored_label = self.label_normalizer.inverse_transform(label)
        error = abs(restored_label - output) / restored_label
        return np.mean(error)

    def tensorboard_loss(self, loss_info, mode):
        """
        #将iteration loss的信息保存到tensorboard
        """
        self.writer.add_scalar('running_loss', loss_info['running_loss'], self.step)
        if mode == 'train':
            self.writer.add_scalar('train_loss', loss_info['running_loss'], self.train_step)
            self.writer.add_scalar('train_rmse', loss_info['rmse'], self.train_step)
            self.writer.add_scalar('train_mape', loss_info['mape'], self.train_step)
        else:
            self.writer.add_scalar('test_loss', loss_info['running_loss'], self.test_step)
            self.writer.add_scalar('test_rmse', loss_info['rmse'], self.test_step)
            self.writer.add_scalar('test_mape', loss_info['mape'], self.test_step)
            self.writer.add_scalar('test_mae', loss_info['mae'], self.test_step)

    def loss_visual(self):
        """
        #画loss图
        """
        if self.args.epochs == 0:
            return
        x = list(int(i[5:]) for i in self.loss_dict.keys())
        df_loss = pd.DataFrame(dict(self.loss_dict)).T.sort_index()
        epoch_train_loss = df_loss['epoch_train_loss'].values.astype(float)
        epoch_test_loss = df_loss['epoch_test_loss'].values.astype(float)
        ###########
        # train_error_rate = df_loss['train_error_rate'].values.astype(float)
        # test_error_rate = df_loss['test_error_rate'].values.astype(float)
        ###########
        train_rmse = df_loss['train_rmse'].values.astype(float)
        test_rmse = df_loss['test_rmse'].values.astype(float)

        train_mape_error = df_loss['train_error'].values.astype(float)
        test_mape_error = df_loss['test_error'].values.astype(float)

        plt.figure()
        plt.subplot(3, 2, 1)
        plt.plot(x, epoch_train_loss, 'bo-', label='epoch_train_loss')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(x, epoch_test_loss, 'bo-', label='epoch_test_loss')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(x, train_rmse, 'bo-', label='train_rmse')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(x, test_rmse, 'bo-', label='test_rmse')
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(x, train_mape_error, 'bo-', label='train_mape_error')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(x, test_mape_error, 'bo-', label='test_mape_error')
        plt.legend()

        plt.savefig(self.args.loss_picture_path + '/' + 'loss.png')
        plt.close('all')
        

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




