# from . import generate_fault_signal_content
from inject import generate_fault_signal_content
import pandas as pd
import numpy as np
from pathlib import Path

def XJTU_inject(signal_cont, file_nums, inject_time, type, p):
    '''
    向XJTU-SY数据集注入故障信号
    signal_cont: 信号所在文件夹
    file_nums: 使用信号文件数量（用文件夹中前file_nums个）
    inject_time: 注入故障信号的时间点
    type: 注入故障信号的类型
    p: 注入故障信号的参数
    '''

    name = Path(signal_cont).name # 信号文件夹名
    save_path = f'data\datasets\GEN\XJTU-SY_{name}_{type}_{p}' # 保存路径
    file_len = 32768 # 信号文件长度
    fz = 25600 # 采样频率
    channel = 1 # 选择通道

    signal_path = Path(signal_cont)
    files = [f for f in signal_path.iterdir() if f.suffix == '.csv']
    files.sort(key=lambda x: int(x.stem))

    dataframes = []

    for file in files[:file_nums]:
        file_path = signal_path / file
        df = pd.read_csv(file_path)
        dataframes.append(df)

    result = pd.concat(dataframes, axis=0).values # shape: (nums*fz, 2)
    normal_signal = result[:,channel].flatten() # shape: (nums*fz,) 
        # 注意这里只用一个通道

    generate_fault_signal_content(normal_signal, inject_time, save_path, file_len, fz=fz, type=type, fault_level_param=p)

'''
注入故障的参数写在params中，每个元素是一个字典，包含以下键值对：
signal_cont: 信号所在文件夹
file_nums: 使用信号文件数量（用文件夹中前file_nums个）
inject_time: 注入故障信号的时间点
type: 注入故障信号的类型
p: 注入故障信号的参数
'''
params=[
    {
        'signal_cont': r'D:\Development\Datasets\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_1',
        'file_nums': 61,
        'inject_time': 30,
        'type': 'Gause',
        'p': 1
    },
    {
        'signal_cont': r'D:\Development\Datasets\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets\40Hz10kN\Bearing3_1',
        'file_nums': 2136,
        'inject_time': 1000,
        'type': 'Gause',
        'p': 1
    },
    {
        'signal_cont': r'D:\Development\Datasets\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets\40Hz10kN\Bearing3_3',
        'file_nums': 305,
        'inject_time': 200,
        'type': 'Gause',
        'p': 1
    },
    {
        'signal_cont': r'D:\Development\Datasets\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets\40Hz10kN\Bearing3_4',
        'file_nums': 915,
        'inject_time': 600,
        'type': 'Gause',
        'p': 1
    },
    {
        'signal_cont': r'D:\Development\Datasets\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets\37.5Hz11kN\Bearing2_1',
        'file_nums': 366,
        'inject_time': 150,
        'type': 'Gause',
        'p': 1
    },
    {
        'signal_cont': r'D:\Development\Datasets\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets\37.5Hz11kN\Bearing2_3',
        'file_nums': 91,
        'inject_time': 5,
        'type': 'Gause',
        'p': 1
    },
]

if __name__ == '__main__':
    for param_dict in params:
        XJTU_inject(**param_dict)
