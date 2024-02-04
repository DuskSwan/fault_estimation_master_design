# encoding: utf-8

import sys
from loguru import logger

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('.')
from config import cfg

from run.tools import signal_to_XY, raw_signal_to_errors
from utils import set_random_seed, initiate_cfg
from data import make_data_loader
from engine.inference import inference

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

    plt.title(cfg.INFERENCE.UNKWON_PATH.split('/')[-1] + ' MAE distribution')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main('./config/XJTU_draw_distr.yml')
