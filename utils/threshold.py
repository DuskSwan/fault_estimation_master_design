import numpy as np

def calc_thresholds(arr,method = ['gause'], p = 0.99, boundary = 3.5):
    thresholds = {}
    for m in method: thresholds[m] = cala_threshold(arr,m,p,boundary)
    return thresholds

def cala_threshold(arr,method = 'gause',p = 0.99, boundary = 3.5):
    # 计算阈值
    if(method == 'gause'): return calc_gause_threshold(arr)
    elif(method == 'log'): return calc_log_threshold(arr)
    elif(method == 'percentile'): return calc_percentile(arr,p)
    elif(method == 'boxplot'): return calc_boxplot_threshold(arr)
    elif(method == 'Z'): return calc_threshold_based_on_Z(arr,boundary)
    else: raise Exception("Unknown method {}".format(method))

def calc_gause_threshold(samples):
    mean = np.mean(samples)
    std = np.std(samples)
    t  = mean + 3 * std
    return t

def calc_percentile(arr,p = 0.99):
    # 计算p-tile，即p分位数
    arr = np.array(arr)
    return np.percentile(arr,p*100)

def calc_boxplot_threshold(samples):
    # 计算 IQR
    q75, q25 = np.percentile(samples, [75 ,25])
    iqr = q75 - q25
    # 计算上边界
    upper_bound = q75 + 1.5 * iqr
    return upper_bound

def calc_log_threshold(samples):
    transformed_samples = np.log(samples)
    # 计算平均值和标准差
    mean = np.mean(transformed_samples)
    std = np.std(transformed_samples)
    # 计算阈值
    t  = mean + 3 * std
    return np.exp(t)

def calc_threshold_based_on_Z(samples, boundary = 3.5):
    # 计算中位数
    median = np.median(samples)
    # 计算 IQR
    q75, q25 = np.percentile(samples, [75 ,25])
    iqr = q75 - q25

    # Z = (x−中位数) / (IQR / 1.349)​ = 1.349 (x−中位数) / IQR >< boundary
    # => 1.349 (x−中位数) >< boundary * IQR
    # => x−中位数 >< boundary * IQR / 1.349
    # => x >< boundary * IQR / 1.349 + 中位数
    t = boundary * iqr / 1.349 + median
    return t