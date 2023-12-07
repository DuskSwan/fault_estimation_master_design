# -*- coding: utf-8 -*-
"""
实现时频域提取特征与预测的接驳
"""
import sys
from pathlib import Path
from loguru import logger
import time

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

import torch.nn.functional as F

sys.path.append('.')

from config import cfg
from utils import set_random_seed
from data import make_data_loader
from engine.example_trainer import do_train
from modeling import build_model
from solver import make_optimizer

set_random_seed(0)

def train(cfg):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    arguments = {}

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        None,
        F.cross_entropy,
    )

def main(priv_cfg_path):
    cfg.merge_from_file(priv_cfg_path)
    cfg.freeze()

    output_dir = Path(cfg.OUTPUT_DIR)
    if not output_dir.exists: output_dir.mkdir()

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    logger.add(str(cfg.LOG_DIR / f'{cur_time}.log'), rotation='1 day', encoding='utf-8')
    logger.info("In device {}".format(cfg.MODEL.DEVICE))

    logger.info("Running with config:\n{}".format(cfg))

    train(cfg)


if __name__ == '__main__':
    main()