# -*- coding: utf-8 -*-
"""
实现时频域提取特征与预测的接驳
"""
import sys
from loguru import logger

sys.path.append('.')
from config import cfg
from utils import set_random_seed,initiate_cfg
from utils.features import view_features_DTW


def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg, extra_cfg_path)

    logger.info("Search propre features...")
    logger.info("Running with config:\n{}".format(cfg))

    # calculate features and rank
    if(cfg.FEATURE.NEED_VIEW):
        ranked_feat = view_features_DTW(cfg)
        logger.info("features ranked:\n{}".format('\n'.join(f"{k}: {v}" for k, v in ranked_feat))) 

if __name__ == '__main__':
    main('config/XJTU_find.yml')