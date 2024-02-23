# encoding: utf-8

import sys
from loguru import logger
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('.')
from config import cfg

from run.tools import signal_to_XY, signal_to_raw_XY
from utils import set_random_seed,initiate_cfg, sheet_cut
from data import make_data_loader
from modeling import build_model
from solver import make_optimizer

from engine.trainer import do_train
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
    
    save_path = 'output/' + Path(cfg.INFERENCE.TEST_CONTENT).stem + '_Base.png'
    plt.savefig(save_path)
    # plt.show()

    logger.info(f'res is {res}')
    logger.info('Finish testing, result saved in {}'.format(save_path))
    return res

def no_features(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg,extra_cfg_path)

    # get data
    normal_path = cfg.TRAIN.NORMAL_PATH
    normal_signal = pd.read_csv(normal_path).values
    normal_X,normal_Y = signal_to_raw_XY(cfg, normal_signal)
    logger.info('X:{} Y:{}'.format(normal_X.shape,normal_Y.shape))
    train_loader = make_data_loader(cfg, normal_X,normal_Y, is_train=True)
    val_loader = None

    # GET MODEL
    _, in_len, in_tunnel = normal_X.shape
    _, out_len, out_tunnel = normal_Y.shape
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

    # calculate threshold
    error_list = inference(cfg, model, train_loader)
    errors = torch.stack(error_list)
    if(cfg.DEVICE != "cpu"): errors = errors.cpu()
    logger.info('Normal signal: Max error {:.4f} , Min error {:.4f}， Mean error {:.4f}'
                .format(errors.max().item(), errors.min().item(), errors.mean().item()))
    errors_arr = errors.numpy()
    thresholds = calc_thresholds(errors_arr, method = cfg.FEATURE.USED_THRESHOLD)
    threshold = thresholds['Z']

    # full roll test
    cont = Path(cfg.INFERENCE.TEST_CONTENT)
    files = sorted(cont.glob('*.csv'), key=lambda x: int(x.stem))
    res = {}
    for file in files:
        logger.info('Start to test file: {}'.format(file.stem))
        signal = pd.read_csv(file).values
        X,Y = signal_to_raw_XY(cfg, signal)
        test_loader = make_data_loader(cfg, X,Y, is_train=False)
        error_list = inference(cfg, model, test_loader)
        errors = torch.stack(error_list)
        if(cfg.DEVICE != "cpu"): errors = errors.cpu()

        logger.info('Current file index: {}'.format(file.stem))
        logger.info('Unkwon signal: Max error {:.4f} , Min error {:.4f}, Mean error {:.4f}'
                    .format(errors.max().item(), errors.min().item(), errors.mean().item()))
        
        unknown_errors_arr = errors.numpy()
        num_greater_than_threshold = (unknown_errors_arr > threshold).sum()
        ratio = num_greater_than_threshold / unknown_errors_arr.size
        res[file.stem] = (num_greater_than_threshold, ratio)
        print("大于阈值的元素数量：", num_greater_than_threshold)
        print("大于阈值的元素比例：", ratio)
    
    # draw result
    x = list(res.keys())
    y = [x[1] for x in res.values()]
    xticks = x[::5]
    # fig = plt.figure()
    plt.plot(x,y, label=cont.stem)
    plt.xticks(xticks, rotation=45)
    plt.ylabel('ratio')
    plt.xlabel('file index')
    plt.title('ratio of mean greater than threshold')
    plt.legend()
    
    save_path = 'output/' + Path(cfg.INFERENCE.TEST_CONTENT).stem + '_NoFeat.png'
    plt.savefig(save_path)
    plt.show()

    # print result
    logger.info('File index, ratio of elements greater than threshold')
    for idx,v in enumerate(res.items()):
        logger.info('{:3d}, {:.4f}'.format(idx, v[1]))

if __name__ == '__main__':
    # baseline('./config/XJTU_test.yml')
    no_features('./config/XJTU_test.yml')
