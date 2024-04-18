import argparse
import json
import os
import sys
import optuna
import math
import warnings
from loguru import logger

import warnings

sys.path.insert(0, '..')
sys.path = list(set(sys.path))
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from trans_train import Train
import logging

#
warnings.filterwarnings("ignore")


def train_optuna(trial):
    params_train = {
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.01, 0.02, 0.03, 0.001, 0.005, 0.0005, 0.0001, 0.00001]),
        'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "cosine_factor": trial.suggest_categorical("cosine_factor",
                                                   [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 1.0]),
        "drop_out": trial.suggest_categorical("drop_out", [0.1, 0.3, 0.4, 0.6, 0.9]),
        "last_kernel_num": trial.suggest_int("last_kernel_num", 3, 15, step=1),
    }
    parser = argparse.ArgumentParser(description='Automatic parameter setting')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(os.getcwd()), 'finetuning/configs/model_config.json'))
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=p_args)

    tr = Train(args)

    train_args, mape, rmse, mae = tr.main(params_train)

    del tr

    return mape


def main(args):
    if args.optuna:

        logger.info("Train with the OPTUNA framework")
        try:
            study = optuna.load_study(study_name=args.study_name, storage='sqlite:///db.sqlite1')
        except:
            study = optuna.create_study(direction="minimize", study_name=args.study_name,
                                        load_if_exists=False, sampler=optuna.samplers.TPESampler(),
                                        storage='sqlite:///db.sqlite1')
        # study.optimize(train_and_evalute, n_trials=100,timeout=600)
        study.optimize(train_optuna, n_trials=args.n_trials)
        best_trial = study.best_trial
        logger.info("best_trial", best_trial)

        for key, value in best_trial.params.items():
            logger.info("%s: %s" % (key, value))


    else:
        Train(args).main

        # 不同fold_num:
        # fold_nums = [0,1,2,3,4]
        # for args.fold_num in fold_nums:
        # Train(args).main()

        """
        #不同训练集大小和文件大小对结果的区别：
        fold_nums = [0,1,2,3,4]
        train_size_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]  # 训练集划分比例
        #file_ratios = [0.25, 0.5, 0.75]  #文件数据划分比例
        for args.train_size_ratio in train_size_ratios:
            #for args.file_ratio in file_ratios:
            for args.fold_num in fold_nums:
                Train(args).main()
        """


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='Automatic parameter setting')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(os.getcwd()), 'finetuning/configs/model_config.json'))
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)

    print("args", args)

    main(args)  # f1
