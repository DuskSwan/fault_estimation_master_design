# -*- coding: utf-8 -*-
"""
实现时频域提取特征与预测的接驳
"""
import sys
import time
from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np
import torch.nn as nn

sys.path.append('.')
from config import cfg

from utils import set_random_seed
from utils.features import view_features_DTW

from run.tools import signal_to_XY
from data import make_data_loader
from modeling import build_model
from solver import make_optimizer

from engine.trainer import do_train
from engine.inference import do_inference

def train(cfg):
    # get data
    X,Y = signal_to_XY(cfg)
    train_loader = make_data_loader(cfg, X,Y, is_train=True)
    val_loader = None

    # GET MODEL
    _, in_len, in_tunnel = X.shape
    _, out_len, out_tunnel = Y.shape
    net_params = {'input_len':in_len,
                'output_len':out_len,
                'input_dim':in_tunnel,
                'output_dim':out_tunnel,}
    model = build_model(cfg, net_params)
    logger.info('Get model with params: {}'.format(net_params))

    # get solver
    optimizer = make_optimizer(cfg, model)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        nn.MSELoss(reduction='mean'),
    )

def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)

    if(extra_cfg_path): logger.info("try to merge from " + extra_cfg_path)
    extra_cfg = Path(extra_cfg_path)
    if extra_cfg.exists() and extra_cfg.suffix == '.yml':
        cfg.merge_from_file(extra_cfg)
    cfg.freeze()

    if(cfg.LOG.OUTPUT_TO_FILE): 
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        logger.add(cfg.LOG.DIR + f'/{cur_time}.log', rotation='1 day', encoding='utf-8')

    output_dir = Path(cfg.OUTPUT_DIR)
    if not output_dir.exists: output_dir.mkdir()

    logger.info("In device {}".format(cfg.DEVICE))
    logger.info("Running with config:\n{}".format(cfg))

    # calculate features and rank
    ranked_feat = view_features_DTW(cfg)
    logger.info("features ranked:\n{}".format('\n'.join(f"{k}: {v}" for k, v in ranked_feat))) 

    # train
    logger.info("feature(s) used:{}".format(', '.join(cfg.FEATURE.USED_F)))
    train(cfg)

if __name__ == '__main__':
    main('./config/debug.yml')