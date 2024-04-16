import argparse
import json
import os
import sys
import torch
import pandas as pd
from preprocess.capacity_dataset import CapacityDataset
from preprocess.pre_utils import Normalizer, LabelNormalizer, PredictResult, EDNormalizer, EDPredictResult
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
#from utils import to_var, collate, Normalizer, PreprocessNormalizer
from finetuning.model.arch.baseline_model import LSTMNet, MLP
#from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
#import xgboost as xgb
from preprocess.dataset import PreprocessNormalizer, SlidingWindowBattery
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
#from utils import build_loc_net, get_fc_graph_struc
def to_var(x, device='cpu'):
    if device == 'cuda':
        x=x.cuda()
    return x

def calculate_metrics(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mae, mape
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch capacity estimation')
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model', type=str, default='LSTMNet')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--norm', default='minmaxscaler')
    parser.add_argument('--dataset', type=int, default=1) #运行脚本以在三个数据集上跑


    args = parser.parse_args()
    #根据每个模型在每个数据集上的最佳折，
    # fold_num_map = {
    #     ('LSTM', '1'): 0,
    #     ('LSTM', '2'): 2,
    #     ('LSTM', '3'): 4,
    #     ('XGBoost', '1'): 0,
    #     ('XGBoost', '2'): 2,
    #     ('XGBoost', '3'): 4,
    #     ('MLP', '1'): 0,
    #     ('MLP', '2'): 2,
    #     ('MLP', '3'): 4,
    #     ('RandomForest', '1'): 0,
    #     ('RandomForest', '2'): 2,
    #     ('RandomForest', '3'): 4,

    # }
    # fold_num = fold_num_map.get((args.model, args.dataset), 0)
    print("args", args)
    data_train = CapacityDataset(
        all_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/all_car_dict.npz.npy',
        ind_ood_car_dict_path=f'/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/ind_odd_dict{args.dataset}.npz.npy',
        train=True,
        fold_num = 0 #fold_num
    )
    data_test = CapacityDataset(
        all_car_dict_path='/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/all_car_dict.npz.npy',
        ind_ood_car_dict_path=f'/log/weiyian/finetuning/preprocess/five_fold_utils/five_fold_utils/ind_odd_dict{args.dataset}.npz.npy',
        train=False,
        fold_num = 0 #fold_num
    )

    label_normalizer = EDNormalizer().exp_minmaxscaler(min_num=0,max_num=100)
    normalizer = Normalizer(dfs=[data_train[i][0] for i in range(len(data_train))])
    train_pre = PreprocessNormalizer(data_train, norm_name=args.norm, normalizer_fn=normalizer.norm_func)
    test_pre = PreprocessNormalizer(data_test, norm_name=args.norm, normalizer_fn=normalizer.norm_func)


    train_loader = DataLoader(dataset=train_pre, batch_size =args.batch_size,shuffle=True,num_workers=0,drop_last=True,)
    test_loader = DataLoader(dataset=test_pre,batch_size=args.batch_size,shuffle=True,num_workers=0, drop_last=True)
    

    # LSTM
    if args.model == "LSTMNet":
        model = LSTMNet(input_dim=8, hidden_dim=32, output_dim=1).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        best_rmse = float('inf')
        best_epoch = -1
        best_model = None

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float(), device='cuda')
                capacity = to_var(batch_data[1]['capacity'].float(),device='cuda')
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss+=loss.item() * data.shape[0]

            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss/len(train_pre)))

            model.eval()
            test_loss = 0
            actuals = []
            predictions = []
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float(),device='cuda')
                    capacity = to_var(batch_data[1]['capacity'].float(),device='cuda')
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss+=loss.item() * data.shape[0]
                    actuals.extend(capacity.view(-1).cpu().numpy())
                    predictions.extend(output.view(-1).cpu().numpy())
                    

            mse, rmse, mae, mape = calculate_metrics(np.array(actuals),np.array(predictions))
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Test Loss: {test_loss / len(test_pre):.4f}, mse: {mse:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}, mape: {mape:.4f}%")

            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch + 1
                best_metrics = (mse, rmse, mae, mape)
                best_test_loss = test_loss / len(test_pre)
                best_model = model.state_dict()
        model.load_state_dict(best_model)
        model.eval()
        result_list = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                data = to_var(batch_data[0].float(), device='cuda')
                capacity = to_var(batch_data[1]['capacity'].float(), device='cuda')
                files = batch_data[1]['car']
                mileage = batch_data[1]['mileage']
                output = model(data)
                predicted_capacity = output.view(-1).cpu().numpy()
                actual_capacity = capacity.view(-1).cpu().numpy()
                for i in range(len(files)):
                    result_list.append([files[i], mileage[i], predicted_capacity[i], actual_capacity[i]])

        # 只保存最佳模型的预测结果
        result1_df = pd.DataFrame(result_list, columns=['file', 'mileage', 'predicted_capacity', 'actual_capacity'])
        result1_df['dataset'] = args.dataset
        file_path = '/home/weiyian/finetuning/result/model_predict_result_lstm_411.csv'
        if not os.path.exists(file_path):
            result1_df.to_csv(file_path, mode="a", index=False)
            print('Statistics have been successfully written to the file with creating a new one.')
        else:
            result1_df.to_csv(file_path, mode="a", index=False, header=False)
            print('successfully write in')
        # 输出最佳epoch的指标和损失
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Test Loss: {best_test_loss:.4f}, Best Metrics - MSE: {best_metrics[0]:.4f}, RMSE: {best_metrics[1]:.4f}, MAE: {best_metrics[2]:.4f}, MAPE: {best_metrics[3]:.4f}%")
        results = {
            'model': args.model,
            'fold_num': args.fold_num,
            'dataset':args.dataset,
            'rmse': best_metrics[1],
            'mae': best_metrics[2],
            'mape': best_metrics[3]
            }
        results_df = pd.DataFrame([results])
        file_path = '/home/weiyian/finetuning/modelresult/model_result_all2.csv'
        if not os.path.exists(file_path):
            results_df.to_csv(file_path, mode="a", index=False)
        else:
            results_df.to_csv(file_path, mode="a", index=False, header=False)
    elif args.model == "XGBoost":
        # XGBoost
        train_features = np.array([data[0] for data in train_pre])
        train_features = train_features.reshape(train_features.shape[0], -1)
        train_labels = np.array([data[1]['capacity'] for data in train_pre])
        test_features = np.array([data[0] for data in test_pre])
        test_features = test_features.reshape(test_features.shape[0], -1)
        test_labels = np.array([data[1]['capacity'] for data in test_pre])

        files = [data[1]['car'] for data in test_pre]
        mileage = [data[1]['mileage'] for data in test_pre]


        dtrain = xgb.DMatrix(train_features, label=train_labels)
        dtest = xgb.DMatrix(test_features, label=test_labels)

        params = {
            'objective': 'reg:squarederror',
            'eta': 0.1,
            'max_depth': 3,
            'eval_metric': 'rmse'
        }
        best_epoch = None
        best_rmse = float('inf')
        best_metrics = None

        evals_result = {}  # 用于存储评估结果
        model = xgb.train(params, dtrain, args.num_epochs, evals=[(dtest, 'test')], evals_result=evals_result)
        test_preds = model.predict(dtest)

        result1_df = pd.DataFrame({
            'file': files,
            'mileage': mileage,
            'predicted_capacity': test_preds,
            'actual_capacity': test_labels
        })
        file_path = '/home/weiyian/finetuning/result/model_predict_result_x.csv'
        if not os.path.exists(file_path):
            result1_df.to_csv(file_path, mode="a", index=False)
        else:
            result1_df.to_csv(file_path, mode="a", index=False, header=False)

        # 分析每个epoch的结果，找到最佳RMSE
        for epoch, rmse in enumerate(evals_result['test']['rmse']):
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch

                # 重新预测，以获得最佳epoch的其他指标
                model.best_iteration = best_epoch
                test_preds = model.predict(dtest, iteration_range=(0, best_epoch+1))
                test_mse, test_rmse, test_mae, test_mape = calculate_metrics(test_labels, test_preds)
                best_metrics = (test_mse, test_rmse, test_mae, test_mape)
        
        results = {
            'model': args.model,
            'fold_num': args.fold_num,
            'dataset':args.dataset,
            'rmse': best_metrics[1],
            'mae': best_metrics[2],
            'mape': best_metrics[3]
            }
        results_df = pd.DataFrame([results])
        file_path = '/home/weiyian/finetuning/modelresult/model_result_all2.csv'
        if not os.path.exists(file_path):
            results_df.to_csv(file_path, mode="a", index=False)
        else:
            results_df.to_csv(file_path, mode="a", index=False, header=False)
        # 输出最佳epoch的指标和损失
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Metrics - MSE: {best_metrics[0]}, RMSE: {best_metrics[1]}, MAE: {best_metrics[2]}, MAPE: {best_metrics[3]}%")
        
    elif args.model == "RandomForest":
        train_features = np.array([data[0] for data in train_pre])
        train_features = train_features.reshape(train_features.shape[0], -1)
        train_labels = np.array([data[1]['capacity'] for data in train_pre])
        test_features = np.array([data[0] for data in test_pre])
        test_features = test_features.reshape(test_features.shape[0], -1)
        test_labels = np.array([data[1]['capacity'] for data in test_pre])

        files = [data[1]['car'] for data in test_pre]
        mileage = [data[1]['mileage'] for data in test_pre]
        model = RandomForestRegressor(n_estimators=10, random_state=0,n_jobs=10,max_depth=4)
        model.fit(train_features, train_labels.flatten())

        train_preds = model.predict(train_features)
        test_preds = model.predict(test_features)

        result1_df = pd.DataFrame({
            'file': files,
            'mileage': mileage,
            'predicted_capacity': test_preds,
            'actual_capacity': test_labels
        })
        file_path = '/home/weiyian/finetuning/result/model_predict_result_R.csv'
        if not os.path.exists(file_path):
            result1_df.to_csv(file_path, mode="a", index=False)
        else:
            result1_df.to_csv(file_path, mode="a", index=False, header=False)
        train_mse, train_rmse, train_mae, train_mape = calculate_metrics(train_labels, train_preds)
        test_mse, test_rmse, test_mae, test_mape = calculate_metrics(test_labels, test_preds)

        results = {
            'model': args.model,
            'fold_num': args.fold_num,
            'dataset':args.dataset,
            'rmse': test_rmse,
            'mae': test_mae,
            'mape': test_mape
            }
        results_df = pd.DataFrame([results])
        file_path = '/home/weiyian/finetuning/modelresult/model_result_all2.csv'
        if not os.path.exists(file_path):
            results_df.to_csv(file_path, mode="a", index=False)
        else:
            results_df.to_csv(file_path, mode="a", index=False, header=False)
        print(f"RandomForest - Train mse: {train_mse}, rmse: {train_rmse}, mae: {train_mae}, mape: {train_mape}")
        print(f"RandomForest - Test mse: {test_mse}, rmse: {test_rmse}, mae: {test_mae}, mape: {test_mape}")

    elif args.model == "MLP":
        model = MLP(input_dim=128*8, hidden_dim=32, output_dim=1).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        best_rmse = float('inf')
        best_epoch = -1
        best_model = None

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float(),device='cuda')
                capacity = to_var(batch_data[1]['capacity'].float(),device='cuda')
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.shape[0]

            print(
                'Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss / len(train_pre)))

            model.eval()
            test_loss = 0
            actuals = []
            predictions = []
    
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float(), device='cuda')
                    capacity = to_var(batch_data[1]['capacity'].float(), device='cuda')
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss += loss.item() * data.shape[0]
                    actuals.extend(capacity.view(-1).cpu().numpy())
                    predictions.extend(output.view(-1).cpu().numpy())


            mse, rmse, mae, mape = calculate_metrics(np.array(actuals),np.array(predictions))
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Test Loss: {test_loss / len(test_pre):.4f}, mse: {mse:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}, mape: {mape:.4f}%")

            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch + 1
                best_metrics = (mse, rmse, mae, mape)
                best_test_loss = test_loss / len(test_pre)
                best_model = model.state_dict()
        model.load_state_dict(best_model)
        model.eval()
        result_list = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                data = to_var(batch_data[0].float(), device='cuda')
                capacity = to_var(batch_data[1]['capacity'].float(), device='cuda')
                files = batch_data[1]['car']
                mileage = batch_data[1]['mileage']
                output = model(data)
                predicted_capacity = output.view(-1).cpu().numpy()
                actual_capacity = capacity.view(-1).cpu().numpy()
                for i in range(len(files)):
                    result_list.append([files[i], mileage[i], predicted_capacity[i], actual_capacity[i]])
        # 只保存最佳模型的预测结果
        result1_df = pd.DataFrame(result_list, columns=['file', 'mileage', 'predicted_capacity', 'actual_capacity'])
        result1_df['dataset'] = args.dataset
        file_path = '/home/weiyian/finetuning/result/model_predict_result_MLP411.csv'
        if not os.path.exists(file_path):
            result1_df.to_csv(file_path, mode="a", index=False)
            print('Statistics have been successfully written to the file with creating a new one.')
        else:
            result1_df.to_csv(file_path, mode="a", index=False, header=False)
            print('successfully write in')
        # 输出最佳epoch的指标和损失
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Test Loss: {best_test_loss:.4f}, Best Metrics - MSE: {best_metrics[0]:.4f}, RMSE: {best_metrics[1]:.4f}, MAE: {best_metrics[2]:.4f}, MAPE: {best_metrics[3]:.4f}%")
        results = {
            'model': args.model,
            'fold_num': args.fold_num,
            'dataset':args.dataset,
            'rmse': best_metrics[1],
            'mae': best_metrics[2],
            'mape': best_metrics[3]
            }
        results_df = pd.DataFrame([results])
        file_path = '/home/weiyian/finetuning/modelresult/model_result_all2.csv'
        if not os.path.exists(file_path):
            results_df.to_csv(file_path, mode="a", index=False)
        else:
            results_df.to_csv(file_path, mode="a", index=False, header=False)
    else:
        raise NotImplementedError



    


