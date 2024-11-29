# from . import generate_fault_signal_content
from inject import generate_fault_signal_content
import pandas as pd
import numpy as np
from pathlib import Path

def XJTU_inject(signal_cont, file_nums, inject_time, type, p):
    # signal_cont = r'D:\Development\Datasets\XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_1'
    # file_nums = 61
    # inject_time = 30
    # type = 'Gause'
    # p = 1

    name = Path(signal_cont).name
    save_path = f'data\datasets\GEN\XJTU-SY_{name}_{type}_{p}'
    file_len = 32768
    fz = 25600

    signal_path = Path(signal_cont)
    files = [f for f in signal_path.iterdir() if f.suffix == '.csv']
    files.sort(key=lambda x: int(x.stem))

    dataframes = []

    for file in files[:file_nums]:
        file_path = signal_path / file
        df = pd.read_csv(file_path)
        dataframes.append(df)

    result = pd.concat(dataframes, axis=0).values # shape: (nums*fz, 2)
    normal_signal = result[:,1].flatten() # shape: (nums*fz,) 
        # 注意这里只用一个通道

    generate_fault_signal_content(normal_signal, inject_time, save_path, file_len, fz=fz, type=type, fault_level_param=p)

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
    }
]

if __name__ == '__main__':
    for param_dict in params:
        XJTU_inject(**param_dict)
