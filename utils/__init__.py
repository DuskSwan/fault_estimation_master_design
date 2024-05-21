import os
import random
import time

from pathlib import Path
from loguru import logger

import numpy as np
import torch


def set_random_seed(seed=31415):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def sheet_cut(x, sublen = None, piece = 500, method = 0, show_para = False):
    '''
    把一个二维数组x切分成很多样本(按照指定的piece或者sublen)，返回三维数组
    method指定了切分的方法：
        0代表平分，每个样本之间没有重叠部分，指定sublen与piece之一即可，优先考虑piece
        1代表均匀的滑窗，取出足够多的样本，样本之间可能有重叠，sublen与piece都要用
        2代表连续的取样本，每个样本不重叠，首尾相接，从左端开始截取，sublen与piece都要用
        3代表逐点取样本，从左端开始截取，每次取出的样本相差1个点，sublen与piece都要用
    show_para指定是否打印参数
    '''
    n,_ = x.shape
    if(method==0): 
        if(sublen is None): 
            sublen = n//piece
            if(show_para): print(f"未指定sublen，计算得{n}//{piece}={sublen}")
        else:
            piece = n//sublen
            if(show_para): print(f"指定sublen={sublen}，n={n}，则piece={piece}")
        start_index = np.arange(0,n,sublen)
        if(n % sublen != 0): start_index = start_index[:-1] #不整除则删除最后一个，避免下标越界
    elif(method==1):
        assert (n-sublen+1>piece), 'piece({}),sublen({}),n({}) don’t match'.format(piece,sublen,n)
        start_index = np.linspace(0, n-sublen-1, piece).astype('int') #记得换成整数索引
    elif(method==2):
        assert sublen*piece <=n, 'piece({}),sublen({}) too big, n({}) too small'.format(piece,sublen,n)
        start_index = np.arange(0,n,sublen)[:piece]
    elif(method==3):
        assert (n-sublen+1>piece), 'piece({})+sublen({}) should less-equ than n({})+1'.format(piece,sublen,n)
        start_index = np.arange(piece)
    
    res = []
    for si in start_index:
        res.append(x[si:si+sublen,:])
    return np.array(res)

def initiate_cfg(cfg,merge_file = ''):
    '''
    Initiate the cfg object with the default config file and the extra config file. 
    The cfg will be frozen after initiation.
    '''
    if(merge_file): logger.info("Try to merge from {}.".format(merge_file))
    else: logger.info("No extra config file to merge.")

    extra_cfg = Path(merge_file)
    if extra_cfg.exists() and extra_cfg.suffix == '.yml':
        cfg.merge_from_file(extra_cfg)
        logger.info("Merge successfully.")
    else:
        logger.info("Wrong file path or file type of extra config file.")

    cfg.DESIGN.FSUBLEN = cfg.DESIGN.M + cfg.DESIGN.P
    cfg.DESIGN.PIECE = cfg.DESIGN.FPIECE * (cfg.DESIGN.M + cfg.DESIGN.P)

    cfg.freeze()

    if(cfg.LOG.OUTPUT_TO_FILE): 
        logger.info("Output to file.")
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        logger.add(cfg.LOG.DIR + f'/{cfg.LOG.PREFIX}_{cur_time}.log', rotation='1 day', encoding='utf-8')
    else: logger.info("Output to console.")

def sort_list(path_list):
    try:
        # 尝试将元素转换为整数进行排序
        return sorted(path_list, key=lambda p: int(p.name.rstrip('.csv')))
    except ValueError:
        # 如果转换失败，按照字典序进行排序
        return sorted(path_list, key=lambda p: p.name)
    except Exception as e:
        raise e