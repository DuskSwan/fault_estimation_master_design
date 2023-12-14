# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys
from pathlib import Path
from loguru import logger

import torch
import numpy as np

sys.path.append('.')
from config import cfg

from run.tools import signal_to_XY
from utils import set_random_seed
from data import make_data_loader
from engine.inference import inference
# from modeling import build_model


def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)

    if(extra_cfg_path): logger.info("try to merge from " + extra_cfg_path)
    extra_cfg = Path(extra_cfg_path)
    if extra_cfg.exists() and extra_cfg.suffix == '.yml':
        cfg.merge_from_file(extra_cfg)
    cfg.freeze()

    logger.info('Start inference')

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

    # calculate unknown signal MAE
    logger.info('Start to calculate unknown signal MAE...')
    X,Y = signal_to_XY(cfg, is_train=False)
    test_loader = make_data_loader(cfg, X,Y, is_train=False)
    error_list = inference(cfg, model, test_loader)
    errors = torch.stack(error_list)
    if(cfg.DEVICE != "cpu"): errors = errors.cpu()
    logger.info('Max error {:.6f} , Min error {:.6f}, Mean error {}'
                .format(errors.max().item(), errors.min().item(), errors.mean().item()))
    indicator = errors.mean()

    # result
    logger.info('Threshold is {:.6f}'.format(threshold))
    logger.info('Indicator is {:.6f}'.format(indicator))
    if(indicator > threshold): logger.info('Unknown signal is FAULTY.')
    else: logger.info('Unknown signal is NORMAL.')
    


if __name__ == '__main__':
    main()
