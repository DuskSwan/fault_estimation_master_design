import sys
from loguru import logger
from pathlib import Path

import numpy as np
import pandas as pd
from torch import stack as tstack
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import matplotlib.dates as mdates

sys.path.append('.')
from utils import  sheet_cut
from utils.features import signal_to_features_tf
from utils.denoise import array_denoise

from modeling import build_model
from solver import make_optimizer
from data import make_data_loader

from engine.trainer import do_train
from engine.inference import inference
from typing import Tuple

def signal_to_raw_XY(cfg, signal:np.ndarray, show_para=False) -> Tuple[np.ndarray, np.ndarray]:
    samples = sheet_cut(signal,
                        piece=cfg.DESIGN.FPIECE,
                        sublen=cfg.DESIGN.FSUBLEN,
                        method=3,
                        show_para = show_para)
    # make XY
    logger.info('start to make X and Y...')
    X = []
    Y = []
    m = cfg.DESIGN.M
    for signal in samples:
        X.append(signal[:m,:][:])
        Y.append(signal[m:,:][:])
    X,Y = map(np.array,(X,Y))
    return (X,Y)

# signal series (2D) -> signal cuts (3D) -> features seriers (2D) -> features cuts (3D) -> X,Y (3D,3D)
#               sheet_cut      signal_to_features_tf            sheet_cut         feature_cuts_to_XY
def feature_cuts_to_XY(cfg, feature_cuts):
    '''
    features: 3D array， 特征样本集
    return: X,Y, 3D array，3D array
    '''
    logger.info('start to make X and Y...')
    X = []
    Y = []
    m = cfg.DESIGN.M
    for signal in feature_cuts:
        X.append(signal[:m,:][:])
        Y.append(signal[m:,:][:])
    X,Y = map(np.array,(X,Y))
    logger.info('X:{} Y:{}'.format(X.shape,Y.shape))
    return (X,Y)

# signal series (2D) -> ...... -> X,Y (3D,3D)
#                   signal_to_XY
def signal_to_XY(cfg, is_train=True, path = None):
    if(is_train): data_path = cfg.TRAIN.NORMAL_PATH
    elif(path): data_path = path
    else: data_path = cfg.INFERENCE.UNKWON_PATH

    signal = pd.read_csv(data_path).values #读成numpy数组
    assert cfg.DATA.USED_CHANNELS, 'cfg.DATA.USED_CHANNELS should not be empty, check chanmels selected'
    assert max(cfg.DATA.USED_CHANNELS) < signal.shape[1], 'cfg.DATA.USED_CHANNELS should be less than signal channels'
    signal = signal[:,cfg.DATA.USED_CHANNELS] #只取需要的通道
    n,_ = signal.shape # 1958912
    piece = cfg.DESIGN.PIECE
    subl = cfg.DESIGN.SUBLEN

    # 去噪
    if(cfg.DENOISE.NEED):
        logger.info('start to denoise...')
        signal = array_denoise(signal, method=cfg.DENOISE.METHOD, step=cfg.DENOISE.SMOOTH_STEP, wavelet=cfg.DENOISE.WAVELET, level=cfg.DENOISE.LEVEL)
    
    # 判断采样点数是否足够
    assert n > piece + subl, f"原始采样点数为{n}，所需采样点数为{piece}+{subl}={piece+subl}，无法达标"
    
    # 特征提取
    logger.info('start to extract features...')
    signal_cuts = sheet_cut(signal, subl, piece, method = 3, show_para = True)
        # signal_cuts是时间序列切片得到的样本集，是三维数组（单个样本是二维数组），每个样本计算一个向量作为特征
        # 3意味着这一步需要保证特征是连续提取的
    assert cfg.FEATURE.USED_F, 'cfg.FEATURE.USED_F should not be empty'
    features = signal_to_features_tf(signal_cuts,feat_func_name_list = cfg.FEATURE.USED_F) 
        # 形状(piece,p_feature)
    logger.info('features shape: {}'.format(features.shape))

    # 特征划分
    feature_cuts = sheet_cut(features, sublen=cfg.DESIGN.FSUBLEN, piece=cfg.DESIGN.FPIECE, method=1) 
        #再次切分成小数据集，现在是三维数组

    return feature_cuts_to_XY(cfg, feature_cuts)

def raw_signal_to_errors(cfg, model, is_normal=True, file_path=None) -> np.ndarray:
    logger.info('Start to calculate error scores...(is_normal={})'.format(is_normal))

    if(is_normal): X,Y = signal_to_XY(cfg, is_train=True)
    else: X,Y = signal_to_XY(cfg, is_train=False, path=file_path)

    loader = make_data_loader(cfg, X,Y, is_train=False)
    error_list = inference(cfg, model, loader)
    errors = tstack(error_list).cpu().numpy()
    logger.info('Max error {:.4f} , Min error {:.4f}， Mean error {:.4f}'
                .format(errors.max(), errors.min(), errors.mean()))
    return errors

def set_train_model(cfg):
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

    train_losses,val_losses = do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        nn.MSELoss(reduction='mean'),
    )

    return model,train_losses,val_losses

from datetime import datetime

def date_format(date_str):
    # Convert the string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    # Format the datetime object as "date-month-year hour:min:second"
    formatted_date = date_obj.strftime("%d-%m-%Y %H:%M:%S")
    return formatted_date
    

def plot_time_series(cfg, series: pd.Series, suffix='ErrRatio'):

    logger.info('Start to plot time series \n {}'.format(series))

    # 尝试检测索引类型（时间戳或整数）
    try:
        # 尝试将索引转换为DatetimeIndex
        temp_index = pd.to_datetime(series.index, format='%Y.%m.%d.%H.%M.%S', errors='raise')
        # 如果成功，假设是时间戳
        is_timestamp = True
    except ValueError:
        # 转换失败，假设是整数
        is_timestamp = False

    if is_timestamp:
        # 使用日期格式化
        x = temp_index
        plt.gca().xaxis.set_major_formatter(DateFormatter("%d-%m-%Y"))
        plt.gca().xaxis.set_major_locator(AutoDateLocator())
    else:
        # 索引被假设为整数，直接使用
        x = pd.RangeIndex(start=0, stop=len(series))
    
    cont = Path(cfg.INFERENCE.TEST_CONTENT)
    y = series.values
    interval = len(x) // 10 if len(x) >=50 else 1
    xticks = x[::interval] if not is_timestamp else x[::interval].strftime("%d-%m-%Y")
    
    import matplotlib
    matplotlib.use('TkAgg')
    plt.figure(figsize=(18,15))

    if(cfg.DENOISE.SHOW_TYPE == 'original only'):
        plt.plot(x, y)
        idx = np.argmax(y > cfg.INFERENCE.MAE_ratio_threshold)
        plt.scatter(x[idx], y[idx], c='r', label='MAE ratio > threshold')
        if is_timestamp:
            date = str(x[idx])
            scat_text = f'({date_format(date)})'
        else:
            scat_text = f'({idx})'
        plt.text(x[idx], y[idx], scat_text, ha='right', color='r')
    elif(cfg.DENOISE.SHOW_TYPE == 'denoised only'):
        denoised_y = array_denoise(y, 
                                   method=cfg.DENOISE.SHOW_METHOD, 
                                   step=cfg.DENOISE.SHOW_SMOOTH_STEP, 
                                   wavelet=cfg.DENOISE.SHOW_WAVELET, 
                                   level=cfg.DENOISE.SHOW_LEVEL)
        denoised_y = np.clip(denoised_y, 0, 1)
        plt.plot(x, denoised_y)
        idx = np.argmax(denoised_y > cfg.INFERENCE.MAE_ratio_threshold)
        plt.scatter(x[idx], denoised_y[idx], c='r', label='MAE ratio > threshold')
        if is_timestamp:
            date = str(x[idx])
            scat_text = f'({date_format(date)})'
        else:
            scat_text = f'({idx})'
        plt.text(x[idx], denoised_y[idx], scat_text, ha='right', color='r')
    elif(cfg.DENOISE.SHOW_TYPE == 'both'):
        plt.plot(x, y, label='original')
        denoised_y = array_denoise(y, 
                                   method=cfg.DENOISE.SHOW_METHOD, 
                                   step=cfg.DENOISE.SHOW_SMOOTH_STEP, 
                                   wavelet=cfg.DENOISE.SHOW_WAVELET, 
                                   level=cfg.DENOISE.SHOW_LEVEL)
        denoised_y = np.clip(denoised_y, 0, 1)
        plt.plot(x, denoised_y, label='denoised')

        idx1 = np.argmax(y > cfg.INFERENCE.MAE_ratio_threshold)
        plt.scatter(x[idx1], y[idx1], label='original ratio > threshold')
        plt.text(x[idx1], y[idx1], f'({x[idx1]})', color='r',
                 ha='right')

        idx2 = np.argmax(denoised_y > cfg.INFERENCE.MAE_ratio_threshold)
        plt.scatter(x[idx2], denoised_y[idx2], label='denoised MAE ratio > threshold')
        plt.text(x[idx2], denoised_y[idx2], f'({x[idx2]})', color='r',
                 ha='right')
    else:
        raise ValueError('cfg.DENOISE.SHOW_TYPE should be one of "original only", "denoised only", "both"')
    plt.axhline(y=cfg.INFERENCE.MAE_ratio_threshold, linestyle='--', color='r', label='threshold')

    # plt.ylim(0, 1)
    plt.xlabel('Time' if is_timestamp else 'Index')
    # plt.xticks(xticks,rotation=45)
    plt.title(cont.stem + ' ratio of elements greater than threshold')  # 假设cont是一个Path对象
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

    save_path = f'output/{cont.stem}_{suffix}.png'
    plt.savefig(save_path)
    # plt.show()

    logger.info(f'Plot saved at {save_path}')
    return save_path

def save_arraylike(arraylike, cont, name):
    save_cont = Path(cont)
    if not save_cont.exists():
        save_cont.mkdir(parents=True, exist_ok=True)
    if(isinstance(arraylike, np.ndarray)):
        path = save_cont / f'{name}.npy'
        np.save(path, arraylike)
    elif(isinstance(arraylike, pd.Series)):
        path = save_cont / f'{name}.csv'
        arraylike.to_csv(path)
    else:    
        raise ValueError('arraylike should be np.ndarray or pd.Series')
