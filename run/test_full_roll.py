# encoding: utf-8

import sys
from loguru import logger
from pathlib import Path

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append('.')
from config import cfg

from run.tools import signal_to_XY
from utils import set_random_seed,initiate_cfg
from data import make_data_loader
from engine.inference import inference

from utils.threshold import calc_thresholds

def full_roll_test(cfg, model, threshold):
    cont = Path(cfg.INFERENCE.TEST_CONTENT)
    files = sorted(cont.glob('*.csv'), key=lambda x: int(x.stem))
    res = {}
    for file in files:
        logger.info('Start to test file: {}'.format(file.stem))
        X,Y = signal_to_XY(cfg, is_train=False, path = file)
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
    
    plt.plot(res.keys(), [x[1] for x in res.values()])
    plt.title('Ratio of elements greater than threshold')
    plt.legend()
    plt.show()

    return res

def main(extra_cfg_path = ''):
    set_random_seed(cfg.SEED)
    initiate_cfg(cfg,extra_cfg_path)

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
    logger.info('Normal signal: Max error {:.4f} , Min error {:.4f}， Mean error {:.4f}'
                .format(errors.max().item(), errors.min().item(), errors.mean().item()))
    errors_arr = errors.numpy()
    thresholds = calc_thresholds(errors_arr, method = cfg.FEATURE.USED_THRESHOLD)
    threshold = thresholds['Z']

    # full roll test
    res = full_roll_test(cfg, model, threshold)

    # result
    logger.info('File index, ratio of elements greater than threshold')
    for idx,(num, ratio) in enumerate(res.items()):
        logger.info('{:3d}, {:.4f}'.format(idx, ratio))
    

if __name__ == '__main__':
    main('./config/XJTU_test.yml')
