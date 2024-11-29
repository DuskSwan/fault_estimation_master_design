# encoding: utf-8
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inject_fault(normal_signal, inject_time, fz=3000, type='Gause', fault_level_param=1):
    '''
    用于向正常信号中注入故障，返回注入故障后的信号。注入故障的信号结束意味着因为故障而停机
    input
        normal_signal: numpy array, 正常信号，长度为n，一维数组
        inject_time: int, 开始注入的时间（单位为秒），需要结合采样频率fz计算出对应的位置
        fz: int, 信号的采样频率
        type: str, 注入故障的类型，'Gause'表示加入逐渐增大的白噪声，'bias'表示加入逐渐增大的偏移量
        fault_level_param: float, 故障的强度参数，数值越大，最终的故障越严重
    output
        fault_signal: numpy array, 注入故障后的信号，长度为n，一维数组
        fault_index: int, 故障发生的位置
    '''
    n = len(normal_signal)
    fault_index = inject_time * fz
    print(f'Total time of signal: {n/fz:.2f}, inject time: {inject_time}, Total signal len: {n}, fault index: {fault_index}')
    assert fault_index < n, 'Inject time is too late'
    fault_len = n - fault_index

    if type == 'Gause':
        noise_std = np.linspace(0, fault_level_param, fault_len)  # 噪声标准差逐渐增大
        noise = np.random.normal(0, noise_std)
    elif type == 'bias':
        noise = np.linspace(0, fault_level_param, fault_len)  # 偏移量逐渐增大
    else:
        raise ValueError('Unknown injected fault type {}'.format(type))

    fault_signal = np.concatenate([normal_signal[:fault_index], normal_signal[fault_index:] + noise])
    return fault_signal, fault_index

def generate_fault_signal_content(normal_signal, inject_time, save_path, file_len, **kwargs):
    '''
    用于生成注入故障后的信号，并保存到文件夹中
    input
        normal_signal: numpy array, 正常信号，长度为n，一维数组
        inject_time: int, 开始注入的时间（单位为秒），需要结合采样频率fz计算出对应的位置
        save_path: str, 保存文件夹的路径
        file_len: int, 每一个文件（信号段）的长度
        kwargs: dict, 传递给inject_fault函数的参数
    '''
    n = len(normal_signal)
    print(f'Generate fault signal content, siganl length is {n} file length is {file_len}')

    fault_signal,fault_index = inject_fault(normal_signal, inject_time, **kwargs)
    fault_signal_len = len(fault_signal)
    fault_signal_num = fault_signal_len // file_len

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for i in range(fault_signal_num):
        fault_signal_piece = fault_signal[i*file_len:(i+1)*file_len]
        csv_name = save_path / f'{i}.csv'
        np.savetxt(csv_name, fault_signal_piece, delimiter=',')

    print(f'Save {fault_signal_num} files to {save_path}, each file length is {file_len} and the last file length is {fault_signal_len % file_len}')
    total_time = fault_signal_len / kwargs['fz']
    file_time = file_len / kwargs['fz']
    print(f'Total time of signal: {total_time:.2f}, each file time: {file_time:.2f}')
    print(f'Fault index: {fault_index}, fault time: {fault_index / kwargs["fz"]:.2f}, fault emerge in file {fault_index // file_len}')

    fig = plt.figure()
    plt.plot(normal_signal, alpha=0.5, label='Normal signal')
    plt.plot(fault_signal, alpha=0.5, label='Fault signal')
    plt.axvline(fault_index, color='r', linestyle='--', label='Fault emerge')
    plt.legend()
    plt.savefig(save_path / 'signal.png')
    plt.close(fig)


def test():
    # 用sin函数模拟正常信号，注入高斯噪声
    
    normal_signal = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
    inject_time = 5
    fz = 100
    fault_signal, fault_index = inject_fault(normal_signal, inject_time, fz, type='Gause', fault_level_param=1)
    plt.plot(normal_signal)
    plt.plot(fault_signal)
    plt.show()

    save_path = 'output/test_fault_signal'
    file_len = 100
    generate_fault_signal_content(normal_signal, inject_time, save_path, file_len, fz=fz, type='Gause', fault_level_param=1)

def generate_fault_signal():
    # 生成正常信号，注入故障，保存到文件夹中
    total_time = 10*60*60 # 10 hours
    inject_time = 7*60*60 # 7 hours
    fz = 2048
    total_len = total_time * fz
    fault_type = 'Gause'
    level = 1 # max fault level
    files_num = 100

    t = np.linspace(0, total_time, total_len)
    normal_signal = np.sin(2*np.pi*50*t) + np.cos(np.pi*10*t) + np.random.normal(0, 0.1, total_len)
    
    fault_signal, fault_index = inject_fault(normal_signal, inject_time, fz, type=fault_type, fault_level_param=level)

    save_path = f'data/datasets/GEN/{fault_type}_{level}'
    file_len = total_len // files_num
    generate_fault_signal_content(normal_signal, inject_time, save_path, file_len, fz=fz, type=fault_type, fault_level_param=level)

if __name__ == '__main__':
    # test()
    generate_fault_signal()
