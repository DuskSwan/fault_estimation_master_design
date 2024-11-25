# -*- coding: utf-8 -*-

import sys
from loguru import logger
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.multioutput import MultiOutputRegressor as MOR

sys.path.append('.')
from config import cfg

from utils import set_random_seed,initiate_cfg
from utils.threshold import calc_thresholds

from run.tools import signal_to_XY, plot_time_series

def train(cfg):
    # get data
    X_train,Y_train = signal_to_XY(cfg, is_train=True)
    num_samples,_,_ = X_train.shape
    # 为了使用模型，需要将数据转换为二维形式
    X_train_reshaped = X_train.reshape(num_samples, -1)
    Y_train_reshaped = Y_train.reshape(num_samples, -1)
    # 训练模型
    model = RFR(n_estimators=100)
    # model = MOR(GBR(n_estimators=100))
    model.fit(X_train_reshaped, Y_train_reshaped)
    
    return model

def sort_list(path_list):
    try:
        # 尝试将元素转换为整数进行排序
        return sorted(path_list, key=lambda p: int(p.name.rstrip('.csv')))
    except ValueError:
        # 如果转换失败，按照字典序进行排序
        return sorted(path_list, key=lambda p: p.name)
    except Exception as e:
        raise e

def full_roll_test(cfg, model, threshold):
    cont = Path(cfg.INFERENCE.TEST_CONTENT)
    files = sort_list(list(cont.glob('*.csv')))
    res = {}
    for file in files:
        logger.info('Start to test file: {}'.format(file.stem))
        # 测试集数据
        X_test,Y_test = signal_to_XY(cfg, is_train=False, path = file)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        Y_test_reshaped = Y_test.reshape(Y_test.shape[0], -1)
        # 测试集预测
        Y_test_pred = model.predict(X_test_reshaped)
        sample_mae = np.mean(np.abs(Y_test_reshaped - Y_test_pred), axis=1)
        logger.info("Test MAE: {:.4f}".format(np.mean(sample_mae)))
       
        unknown_errors_arr = sample_mae
        num_greater_than_threshold = (unknown_errors_arr > threshold).sum()
        ratio = num_greater_than_threshold / unknown_errors_arr.size
        res[file.stem] = (num_greater_than_threshold, ratio)
        print("大于阈值的元素数量：", num_greater_than_threshold)
        print("大于阈值的元素比例：", ratio)
    
    res_series = pd.Series({k: v[1] for k, v in res.items()})
    plot_time_series(cfg, res_series, suffix='ERT')
    return res

def main(extra_cfg_path = ''):

    set_random_seed(cfg.SEED)
    initiate_cfg(cfg, extra_cfg_path)

    logger.info("In device {}".format(cfg.DEVICE))
    logger.info("Running with config:\n{}".format(cfg))

    # train
    logger.info("feature(s) used: {}".format(', '.join(cfg.FEATURE.USED_F)))
    model = train(cfg)

    # calculate threshold
    logger.info('Start to calculate threshold...')
    # get data
    X_train,Y_train = signal_to_XY(cfg, is_train=True)
    num_samples,_,_ = X_train.shape
    # 为了使用模型，需要将数据转换为二维形式
    X_train_reshaped = X_train.reshape(num_samples, -1)
    Y_train_reshaped = Y_train.reshape(num_samples, -1)
    # 重新计算训练集的预测值和MAE
    Y_train_pred = model.predict(X_train_reshaped)
    train_mae = np.mean(np.abs(Y_train_reshaped - Y_train_pred), axis=1)
    logger.info("Train MAE: {:.4f}".format(np.mean(train_mae)))
    # 计算阈值
    thresholds = calc_thresholds(train_mae, method = cfg.FEATURE.USED_THRESHOLD)
    threshold = thresholds['Z']

    # full roll test
    res = full_roll_test(cfg, model, threshold)

    # result
    logger.info('File index, ratio of elements greater than threshold')
    logger.info(res)
    

if __name__ == '__main__':
    main('./config/IMS_test.yml')