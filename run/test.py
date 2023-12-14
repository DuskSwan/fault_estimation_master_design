# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys
from pathlib import Path
from loguru import logger
from torch import load as tload

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
    model = tload(cfg.INFERENCE.MODEL_PATH)
    model.to(cfg.DEVICE)
    logger.info('Change model to {}'.format(cfg.DEVICE))

    # calculate threshold
    X,Y = signal_to_XY(cfg, is_train=True)
    test_loader = make_data_loader(cfg, X,Y, is_train=False)
    error_list = inference(cfg, model, test_loader)
    print(error_list[0].shape)
    errors = [tensor.mean().item() for tensor in error_list]
    errors = np.array(errors)
    logger.info('errors: {}'.format(errors))

    # calculate MAE
    # error = inference(cfg, model, val_loader)


if __name__ == '__main__':
    main()
