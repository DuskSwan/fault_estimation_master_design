# -*- coding: utf-8 -*-

import sys
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.multioutput import MultiOutputRegressor as MOR

sys.path.append('.')
from config import cfg

from utils import set_random_seed,initiate_cfg
from utils.features import view_features_DTW

from run.tools import signal_to_XY

def train_and_test(cfg):
    # get data
    X_train,Y_train = signal_to_XY(cfg, is_train=True)
    num_samples,_,_ = X_train.shape
    # 为了使用模型，需要将数据转换为二维形式
    X_train_reshaped = X_train.reshape(num_samples, -1)
    Y_train_reshaped = Y_train.reshape(num_samples, -1)
    # 训练模型
    # model = RFR(n_estimators=100)
    model = MOR(GBR(n_estimators=100))
    model.fit(X_train_reshaped, Y_train_reshaped)
    # 测试集数据
    X_test,Y_test = signal_to_XY(cfg, is_train=False)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    Y_test_reshaped = Y_test.reshape(Y_test.shape[0], -1)
    # 测试集预测
    Y_test_pred = model.predict(X_test_reshaped)
    sample_mae = np.mean(np.abs(Y_test_reshaped - Y_test_pred), axis=1)
    # 重新计算训练集的预测值和MAE
    Y_train_pred = model.predict(X_train_reshaped)
    train_mae = np.mean(np.abs(Y_train_reshaped - Y_train_pred), axis=1)
    # 输出训练集和测试集的MAE
    print("Train MAE: {:.4f}".format(np.mean(train_mae)))
    print("Test MAE: {:.4f}".format(np.mean(sample_mae)))
    # 绘制训练集和测试集的MAE分布图
    plt.hist(train_mae, bins=20, label='Train MAE')
    plt.hist(sample_mae, bins=20, label='Test MAE', alpha=0.75)
    plt.title(cfg.INFERENCE.UNKWON_PATH.split('/')[-1] + ' MAE Distribution')
    plt.xlabel("Mean Absolute Error")
    # plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    
def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg, extra_cfg_path)

    logger.info("Running with config:\n{}".format(cfg))

    # calculate features and rank
    if(cfg.FEATURE.NEED_VIEW):
        ranked_feat = view_features_DTW(cfg)
        logger.info("features ranked:\n{}".format('\n'.join(f"{k}: {v}" for k, v in ranked_feat))) 

    # train
    logger.info("feature(s) used: {}".format(', '.join(cfg.FEATURE.USED_F)))
    train_and_test(cfg)

if __name__ == '__main__':
    main('./ERT/ERT_config.yml')