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
from utils import set_random_seed, sheet_cut
from utils.features import view_features_DTW, signal_to_features_tf
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer

def train(cfg):
    # 设置参数
    normal_data = pd.read_csv(cfg.TRAIN.NORMAL_PATH).values #读成numpy数组
    n,_ = normal_data.shape # 1958912
    piece = cfg.DESIGN.PIECE
    subl = cfg.DESIGN.SUBLEN
    
    # 判断采样点数是否足够
    assert n > piece + subl, f"原始采样点数为{n}，所需采样点数为{piece}+{subl}={piece+subl}，无法达标"
    
    # 特征提取
    logger.info('start to extract features...')
    XY = sheet_cut(normal_data, subl, piece, method = 3, show_para = True)
        # XY是时间序列切片得到的样本集，是三维数组（单个样本是二维数组），每个样本计算一个向量作为特征
        # 3意味着这一步需要保证特征是连续提取的
    st = time.time()
    normal_f = signal_to_features_tf(XY,feat_func_name_list = cfg.FEATURE.USED_F) 
        # 形状(piece,p_feature)
    et = time.time()
    logger.info('feature extraction finished with time {:.6f}s.'.format(et-st))
    logger.info('features shape: {}'.format(normal_f.shape))

    # 特征划分
    f_dataset = sheet_cut(normal_f, sublen=cfg.DESIGN.FSUBLEN, piece=cfg.DESIGN.FPIECE, method=1) 
        #再次切分成小数据集，现在是三维数组
    
    #制作X与Y
    logger.info('start to make X and Y...')
    X = []
    Y = []
    m = cfg.DESIGN.M
    for signal in f_dataset:
        X.append(signal[:m,:][:])
        Y.append(signal[m:,:][:])
    X,Y = map(np.array,(X,Y))
    logger.info('X:{} Y:{}'.format(X.shape,Y.shape))

    # GET MODEL
    _, in_len, in_tunnel = X.shape
    _, out_len, out_tunnel = Y.shape
    net_params = {'input_len':in_len,
                'output_len':out_len,
                'input_dim':in_tunnel,
                'output_dim':out_tunnel,}
    model = build_model(cfg, net_params)
    logger.info('Get model with params: {}'.format(net_params))

    # 读取设置
    optimizer = make_optimizer(cfg, model)
    scheduler = None

    train_loader = make_data_loader(cfg, X,Y, is_train=True)
    val_loader = make_data_loader(cfg, X,Y, is_train=False)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        nn.MSELoss(reduction='mean'),
    )

def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)

    if(cfg.LOG.OUTPUT_TO_FILE): 
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        logger.add(cfg.LOG.DIR + f'/{cur_time}.log', rotation='1 day', encoding='utf-8')

    if(extra_cfg_path): logger.info("try to merge from " + extra_cfg_path)
    extra_cfg = Path(extra_cfg_path)
    if extra_cfg.exists() and extra_cfg.suffix == '.yml':
        cfg.merge_from_file(extra_cfg)
    cfg.freeze()

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