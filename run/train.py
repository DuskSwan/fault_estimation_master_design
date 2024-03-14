# -*- coding: utf-8 -*-
"""
实现时频域提取特征与预测的接驳
"""
import sys
import time
from pathlib import Path
from loguru import logger

import torch.nn as nn
from torch import save as tsave

sys.path.append('.')
from config import cfg

from utils import set_random_seed,initiate_cfg
from utils.features import view_features_DTW

from run.tools import signal_to_XY
from data import make_data_loader
from modeling import build_model
from solver import make_optimizer

from engine.trainer import do_train

def train(cfg, save=True):
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

    if(not save): return (model, None)
    
    save_cont = Path(cfg.OUTPUT.MODEL_DIR)
    if not save_cont.exists(): save_cont.mkdir()
    save_path = save_cont / cfg.OUTPUT.MODEL_NAME
    tsave(model, save_path)

    return (model, save_path)

def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg, extra_cfg_path)

    logger.info("In device {}".format(cfg.DEVICE))
    logger.info("Running with config:\n{}".format(cfg))

    # calculate features and rank
    if(cfg.FEATURE.NEED_VIEW):
        ranked_feat = view_features_DTW(cfg)
        logger.info("features ranked:\n{}".format('\n'.join(f"{k}: {v}" for k, v in ranked_feat))) 

    # train
    logger.info("feature(s) used: {}".format(', '.join(cfg.FEATURE.USED_F)))
    train(cfg)

if __name__ == '__main__':
    main('./config/CWRU_test.yml')