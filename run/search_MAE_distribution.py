# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys
from loguru import logger

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('.')
from config import cfg

from run.tools import signal_to_XY
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
    X,Y = signal_to_XY(cfg, is_train=True)
    normal_loader = make_data_loader(cfg, X,Y, is_train=False)
    normal_error_list = inference(cfg, model, normal_loader)
    errors = torch.stack(normal_error_list)
    if(cfg.DEVICE != "cpu"): normal_errors = errors.cpu()
    logger.info('Max error {:.4f} , Min error {:.4f}'
                .format(normal_errors.max().item(), normal_errors.min().item()))
    normal_errors = normal_errors.numpy()
    plt.hist(normal_errors,bins=18,
             color='blue',label='normal signal')

    # calculate unknown signal MAE
    logger.info('Start to calculate unknown signal MAE...')
    X,Y = signal_to_XY(cfg, is_train=False)
    test_loader = make_data_loader(cfg, X,Y, is_train=False)
    error_list = inference(cfg, model, test_loader)
    errors = torch.stack(error_list)
    if(cfg.DEVICE != "cpu"): errors = errors.cpu()
    logger.info('Unkwon signal: Max error {:.4f} , Min error {:.4f}, Mean error {:.4f}'
                .format(errors.max().item(), errors.min().item(), errors.mean().item()))
    plt.hist(errors.numpy(),bins=18,
             color='green',label='unknown signal',alpha=0.75)

    # draw thresholds
    thresholds = calc_thresholds(normal_errors, method = cfg.FEATURE.USED_THRESHOLD)
    colors = ['red', 'blue', 'green', 'orange', 'pink']  # 指定不同的颜色
    for i, (k, t) in enumerate(thresholds.items()):
        plt.axvline(x=t, linestyle='--', color=colors[i], label=k)  # 添加竖线并指定颜色
    plt.legend()

    plt.show()
    

if __name__ == '__main__':
    main('./config/CWRU_draw_distr.yml')
