# encoding: utf-8

import sys
from loguru import logger
from pathlib import Path

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('.')
from config import cfg

from run.tools import raw_signal_to_errors
from utils import set_random_seed, initiate_cfg

from utils.threshold import calc_thresholds


def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg,extra_cfg_path)

    # get model
    model = torch.load(cfg.INFERENCE.MODEL_PATH)
    model.to(cfg.DEVICE)
    logger.info('Change model to {}'.format(cfg.DEVICE))

    # calculate MAEs
    logger.info('Start to calculate threshold and distribution...')
    normal_errors = raw_signal_to_errors(cfg, model, is_normal=True)
    plt.hist(normal_errors,bins=18,
             color='blue',label='normal signal')

    # calculate unknown signal MAE
    logger.info('Start to calculate unknown signal MAE...')
    errors = raw_signal_to_errors(cfg, model, is_normal=False)
    plt.hist(errors,bins=18,
             color='green',label='unknown signal',alpha=0.75)

    # draw thresholds
    thresholds = calc_thresholds(normal_errors, method = cfg.FEATURE.USED_THRESHOLD)
    colors = ['red', 'blue', 'green', 'orange', 'pink']  # 指定不同的颜色
    for i, (k, t) in enumerate(thresholds.items()):
        plt.axvline(x=t, linestyle='--', color=colors[i], label=k)  # 添加竖线并指定颜色

    # draw fault signal indicator
    plt.axvline(x = errors.mean(), linestyle='--', color='black', label='mean MAE (fault)')

    title = f'MAE distribution {Path(cfg.INFERENCE.UNKWON_PATH).stem}({cfg.FEATURE.USED_F[0]})'
    plt.title(title)
    plt.legend()
    plt.show()

    #calculate ration
    threshold = thresholds['Z']
    unknown_errors_arr = errors

    num_greater_than_threshold = (unknown_errors_arr > threshold).sum()
    ratio = num_greater_than_threshold / unknown_errors_arr.size

    print("大于阈值的元素数量：", num_greater_than_threshold)
    print("大于阈值的元素比例：", ratio)
    

if __name__ == '__main__':
    cfg.LOG.OUTPUT_TO_FILE = False
    main('./config/CWRU_test.yml')
