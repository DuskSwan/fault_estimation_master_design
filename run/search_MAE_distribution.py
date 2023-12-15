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


def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg,extra_cfg_path)

    # get model
    model = torch.load(cfg.INFERENCE.MODEL_PATH)
    model.to(cfg.DEVICE)
    logger.info('Change model to {}'.format(cfg.DEVICE))

    # calculate threshold
    logger.info('Start to calculate threshold...')
    X,Y = signal_to_XY(cfg, is_train=True)
    normal_loader = make_data_loader(cfg, X,Y, is_train=False)
    error_list = inference(cfg, model, normal_loader)
    errors = torch.stack(error_list)
    if(cfg.DEVICE != "cpu"): errors = errors.cpu()
    logger.info('Max error {:.6f} , Min error {:.6f}'.format(errors.max().item(), errors.min().item()))
    threshold = errors.mean() + 3 * errors.std()

    plt.hist(errors)
    plt.axvline(x=threshold, color='r', linestyle='--')  # 添加竖线
    plt.show()
    

if __name__ == '__main__':
    main('./config/CWRU_draw_distr(122).yml')
