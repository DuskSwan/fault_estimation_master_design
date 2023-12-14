import sys
from loguru import logger

import numpy as np
import pandas as pd

sys.path.append('.')
from utils import  sheet_cut
from utils.features import signal_to_features_tf

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
def signal_to_XY(cfg, is_train=True):
    if(is_train): data_path = cfg.TRAIN.NORMAL_PATH
    else: data_path = cfg.INFERENCE.INFERENCE.UNKWON_PATH

    signal = pd.read_csv(data_path).values #读成numpy数组
    n,_ = signal.shape # 1958912
    piece = cfg.DESIGN.PIECE
    subl = cfg.DESIGN.SUBLEN
    
    # 判断采样点数是否足够
    assert n > piece + subl, f"原始采样点数为{n}，所需采样点数为{piece}+{subl}={piece+subl}，无法达标"
    
    # 特征提取
    logger.info('start to extract features...')
    signal_cuts = sheet_cut(signal, subl, piece, method = 3, show_para = True)
        # signal_cuts是时间序列切片得到的样本集，是三维数组（单个样本是二维数组），每个样本计算一个向量作为特征
        # 3意味着这一步需要保证特征是连续提取的
    features = signal_to_features_tf(signal_cuts,feat_func_name_list = cfg.FEATURE.USED_F) 
        # 形状(piece,p_feature)
    logger.info('features shape: {}'.format(features.shape))

    # 特征划分
    feature_cuts = sheet_cut(features, sublen=cfg.DESIGN.FSUBLEN, piece=cfg.DESIGN.FPIECE, method=1) 
        #再次切分成小数据集，现在是三维数组

    return feature_cuts_to_XY(cfg, feature_cuts)