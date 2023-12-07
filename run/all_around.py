# -*- coding: utf-8 -*-
"""
实现时频域提取特征与预测的接驳
"""
import sys
from pathlib import Path
from loguru import logger
import time
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append('.')

from config import cfg
from utils import set_random_seed

set_random_seed(0)

def main(priv_cfg_path):
    cfg.merge_from_file(priv_cfg_path)
    cfg.freeze()

    output_dir = Path(cfg.OUTPUT_DIR)
    if not output_dir.exists: output_dir.mkdir()

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    logger.add(str(cfg.LOG_DIR / f'{cur_time}.log'), rotation='1 day', encoding='utf-8')
    logger.info("In device {}".format(cfg.MODEL.DEVICE))

    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    val_loader = make_data_loader(cfg, is_train=False)

    inference(cfg, model, val_loader)


if __name__ == '__main__':
    main()