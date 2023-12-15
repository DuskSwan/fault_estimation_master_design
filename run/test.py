# encoding: utf-8

import sys
from loguru import logger

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
sys.path.append('.')
from config import cfg

from run.tools import signal_to_XY
from utils import set_random_seed,initiate_cfg
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

    # calculate threshold
    logger.info('Start to calculate threshold...')
    X,Y = signal_to_XY(cfg, is_train=True)
    normal_loader = make_data_loader(cfg, X,Y, is_train=False)
    error_list = inference(cfg, model, normal_loader)
    errors = torch.stack(error_list)
    if(cfg.DEVICE != "cpu"): errors = errors.cpu()
    logger.info('Normal signal: Max error {:.4f} , Min error {:.4f}， Mean error {:.4f}'
                .format(errors.max().item(), errors.min().item(), errors.mean().item()))
    errors_arr = errors.numpy()
    thresholds = calc_thresholds(errors_arr, method = cfg.FEATURE.USED_THRESHOLD)
    # logger.info('Thresholds are {}'.format(thresholds))

    # calculate unknown signal MAE
    logger.info('Start to calculate unknown signal MAE...')
    X,Y = signal_to_XY(cfg, is_train=False)
    test_loader = make_data_loader(cfg, X,Y, is_train=False)
    error_list = inference(cfg, model, test_loader)
    errors = torch.stack(error_list)
    if(cfg.DEVICE != "cpu"): errors = errors.cpu()
    logger.info('Unkwon signal: Max error {:.4f} , Min error {:.4f}, Mean error {:.4f}'
                .format(errors.max().item(), errors.min().item(), errors.mean().item()))
    indicator = errors.mean().item()

    # result
    for k,threshold in thresholds.items():
        res = 'Normal' if indicator < threshold else 'Abnormal'
        logger.info('{} threshold is {:.4f}, indicator is {:.4f}'.format(k, threshold, indicator))
        logger.info('   Evaluation result is {}'.format(res))
    

if __name__ == '__main__':
    main('./config/BUCT_test.yml')
