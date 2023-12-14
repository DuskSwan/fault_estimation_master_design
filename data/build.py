# encoding: utf-8

from torch.utils import data
from torch import tensor

def make_data_loader(cfg, X, Y, is_train=True):
    '''
    X是输入，Y是输出，类型都是numpy数组
    '''
    if is_train:
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
    else:
        batch_size = cfg.INFERENCE.BATCH_SIZE
        shuffle = False

    X,Y = map(tensor,(X,Y))
    dataset = data.TensorDataset(X,Y) 
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
