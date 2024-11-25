# encoding: utf-8

import sys
from loguru import logger
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('.')
from config import cfg_debug as cfg

from run.train import train
from run.tools import signal_to_XY
from utils import set_random_seed,initiate_cfg
from data import make_data_loader

def sort_list(path_list):
    try:
        # 尝试将元素转换为整数进行排序
        return sorted(path_list, key=lambda p: int(p.name.rstrip('.csv')))
    except ValueError:
        # 如果转换失败，按照字典序进行排序
        return sorted(path_list, key=lambda p: p.name)
    except Exception as e:
        raise e

def main(extra_cfg_path = ''):

    set_random_seed(cfg.SEED)
    initiate_cfg(cfg, extra_cfg_path)

    logger.info("In device {}".format(cfg.DEVICE))
    logger.info("Running with config:\n{}".format(cfg))

    # train
    logger.info("feature(s) used: {}".format(', '.join(cfg.FEATURE.USED_F)))
    model,_ = train(cfg, save=False)

    # calculate prediction
    logger.info('Start to calculate unknown signal MAE...')
    X,Y = signal_to_XY(cfg, is_train=False)
    test_loader = make_data_loader(cfg, X,Y, is_train=False)
    y_true_all = []
    y_pred_all = []
    
    model.eval()
    for X0, Y0 in test_loader:
        prediction = model(X0)
        y_true_all.extend(Y0.detach().numpy().flatten())
        y_pred_all.extend(prediction.detach().numpy().flatten())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # 计算 WAPE
    absolute_errors = np.abs(y_true_all - y_pred_all)
    sum_absolute_errors = np.sum(absolute_errors)
    sum_true_values = np.sum(np.abs(y_true_all))
    wape = (sum_absolute_errors / sum_true_values) * 100

    print(f"WAPE: {wape}%")

    y_true = Y0.detach().numpy().flatten()
    y_pred = prediction.detach().numpy().flatten()
    errors = np.abs(y_true - y_pred)
    t = range(len(y_true))
    plt.plot(t, y_true, label='True')
    plt.plot(t, y_pred, label='Pred')
    plt.plot(t, errors, label='Error')
    plt.legend()
    plt.title('True, Pred, Error with WAPE: {:.2f}%'.format(wape))
    plt.show()


    

if __name__ == '__main__':
    main('./config/XJTU_test.yml')
    
