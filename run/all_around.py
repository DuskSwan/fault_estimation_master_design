# -*- coding: utf-8 -*-
"""
实现时频域提取特征与预测的接驳
"""
import sys
import time
from pathlib import Path
from loguru import logger

sys.path.append('.')
from config import cfg
from utils import set_random_seed, sheet_cut
from utils.features import view_features_DTW, signal_to_features_tf

from engine.example_trainer import do_train
from modeling import build_model
from solver import make_optimizer

def train(cfg):
    # 设置参数
    normal_data = pd.read_csv(cfg.DATASETS.NORMAL_PATH).values #读成numpy数组
    n,_ = normal_data.shape # 1958912
    piece = cfg.DESIGN.PIECE
    subl = cfg.DESIGN.SUBLEN
    
    # 判断采样点数是否足够
    assert n > piece + subl, f"原始采样点数为{n}，所需采样点数为{piece}+{subl}={piece+subl}，无法达标"
    
    # 特征提取
    XY = sheet_cut(normal_data, subl, piece, method = 3, show_para = True)
        # XY是时间序列切片得到的样本集，是三维数组（单个样本是二维数组），每个样本计算一个向量作为特征
        # 3意味着这一步需要保证特征是连续提取的
    st = time.time()
    normal_f = signal_to_features_tf(XY,feat_func_name_list = cfg.FEATURE.USED_F) 
        # 形状(piece,p_feature)
    et = time.time()
    print(normal_f.shape)
    print('draw time:', et-st)

    # 读取设置
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE
    optimizer = make_optimizer(cfg, model)
    scheduler = None

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
    set_random_seed(0)
    cfg.merge_from_file(priv_cfg_path)
    cfg.freeze()

    output_dir = Path(cfg.OUTPUT_DIR)
    if not output_dir.exists: output_dir.mkdir()

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    logger.add(str(cfg.LOG_DIR / f'{cur_time}.log'), rotation='1 day', encoding='utf-8')
    logger.info("In device {}".format(cfg.MODEL.DEVICE))
    logger.info("Running with config:\n{}".format(cfg))

    # calculate features and rank
    feat_dict = view_features_DTW(cfg)
    logger.info(feat_dict)

    # train
    logger.info("feature(s) used:",cfg.FEATURE.USED_F)
    train(cfg)


if __name__ == '__main__':
    main()