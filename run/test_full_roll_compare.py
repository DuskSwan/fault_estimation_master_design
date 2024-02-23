# encoding: utf-8

import sys
from loguru import logger
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('.')
from config import cfg

from run.tools import signal_to_XY
from utils import set_random_seed,initiate_cfg, sheet_cut
from data import make_data_loader
from engine.inference import inference

from utils.threshold import calc_thresholds

def baseline(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg,extra_cfg_path)

    logger.info('Start to calculate threshold...')
    normal_path = cfg.TRAIN.NORMAL_PATH
    signal = pd.read_csv(normal_path).values #读成numpy数组
    mean = signal.mean(axis=0)
    std = signal.std(axis=0)
    alpha = 3
    threshold = alpha * std
    logger.info(f'Normal signal mean is {mean}, {alpha}σ is {threshold}')

    cont = Path(cfg.INFERENCE.TEST_CONTENT)
    files = sorted(cont.glob('*.csv'), key=lambda x: int(x.stem))

    res = {}
    for file in files:
        logger.info('Start to test file: {}'.format(file.stem))
        s = pd.read_csv(file).values 
        samples = sheet_cut(s, 
                            sublen = cfg.DESIGN.SUBLEN, 
                            show_para = True) # (piece,sublen,dim)
        unknown_means = samples.mean(axis=1) # (piece,dim)
        errs = np.abs(unknown_means - mean) # (piece,dim) - (dim,) -> (piece,dim)
        marks = np.any(errs > threshold, axis=1) # (piece,dim) > (dim,) -> (piece,dim) -> (piece,)

        num_marked = marks.sum()
        ratio = num_marked / marks.shape[0]
        res[file.stem] = (num_marked, ratio)
        print("异常元素数量：", num_marked)
        print("异常元素比例：", ratio)
    
    x = list(res.keys())
    y = [x[1] for x in res.values()]
    xticks = x[::5]
    # fig = plt.figure()
    plt.plot(x,y, label=cont.stem)
    plt.xticks(xticks, rotation=45)
    plt.ylabel('ratio')
    plt.xlabel('file index')
    plt.title(f'ratio of mean greater than {alpha}σ')
    plt.legend()
    
    save_path = 'output/' + Path(cfg.INFERENCE.TEST_CONTENT).stem + '_0.png'
    plt.savefig(save_path)
    # plt.show()

    logger.info(f'res is {res}')
    logger.info('Finish testing, result saved in {}'.format(save_path))
    return res

def no_features(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg,extra_cfg_path)

    

if __name__ == '__main__':
    baseline('./config/XJTU_test.yml')
