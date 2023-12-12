import numpy as np
from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt

def compute_correlaion_with_shift(x,y):
    '''
    给出两个序列x和y，计算他们的相关性。由于二者可能有延迟，实际上计算的是滑动x后，最大的相关性
    x，y均为numpy向量
    '''
    assert len(x) == len(y)
    f1 = np.fft.fft(x)
    f2 = np.fft.fft(np.flipud(y))
    cc = np.real(np.fft.ifft(f1 * f2))
    c = np.fft.fftshift(cc)
    assert len(c) == len(x)
    zero_index = len(x) // 2
    shift = zero_index - np.argmax(c)
    
    y_shifted = np.roll(y, -shift)
    corrcoef = np.corrcoef(x, y_shifted)
    return corrcoef

def cross_correlation_using_fft(x, y):
    f1 = np.fft.fft(x)
    f2 = np.fft.fft(np.flipud(y))
    cc = np.real(np.fft.ifft(f1 * f2))
    return np.fft.fftshift(cc)

def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = len(x) // 2
    shift = zero_index - np.argmax(c)
    return shift

def TimeSeriesSimilarity(s1, s2):
    # DTW算法的实现和对比分析展示
    l1 = len(s1)
    l2 = len(s2)
    # plt.plot(s1, "r", s2, "g")
    # plt.show()
    if np.std(s1)==0 or np.std(s2)==0 : return 0 # 方差为0 说明全部值都一样
    s1 = (s1 - np.mean(s1)) / np.std(s1)
    s2 = (s2 - np.mean(s2)) / np.std(s2)
    paths = np.full((l1 + 1, l2 + 1), np.inf)  # 全部赋予无穷大
    paths[0, 0] = 0
    for i in range(l1):
        for j in range(l2):
            d = s1[i] - s2[j]
            cost = d ** 2
            paths[i + 1, j + 1] = cost + min(paths[i, j + 1], paths[i + 1, j], paths[i, j])

    paths = np.sqrt(paths)
    s = paths[l1, l2]
    return s

def FrechetDistance(ptSetA, ptSetB):
    # 获得点集ptSetA中点的个数n
    n = ptSetA.shape[0]
    # 获得点集ptSetB中点的个数m
    m = ptSetB.shape[0]
    # 计算任意两个点的距离矩阵
    # disMat[i][j]对应ptSetA的第i个点到ptSetB中第j点的距离
    disMat = cdist(ptSetA, ptSetB, metric='euclidean')
    # 初始化消耗矩阵
    costMatrix = np.full((n, m), -1.0)
    # 逐行给消耗矩阵赋值
    # 首先给第一行赋值
    # 然后依次给2,3,4,...,m行赋值
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                # 给左上角赋值
                costMatrix[0][0] = disMat[0][0]
            if i == 0 and j > 0:
                # 给第一行赋值
                costMatrix[0][j] = max(costMatrix[0][j-1], disMat[0][j])
            if i > 0 and j == 0:
                # 给第一列赋值
                costMatrix[i][0] = max(costMatrix[i-1][0], disMat[i][0])
            if i > 0 and j > 0:
                # 给其他赋值
                costMatrix[i][j] = max(min(costMatrix[i-1][j],
                                           costMatrix[i-1][j-1],
                                           costMatrix[i][j-1]), disMat[i][j])
    return costMatrix[n-1][m-1]

if __name__ == '__main__':
    # 你的数据
    arr1 = np.random.rand(200)
    arr2 = np.random.rand(200)
    arr3 = arr1 *10
    arr4 = arr1 + 10
    
    corrcoef = compute_correlaion_with_shift(arr1, arr2)[0,1]
    print('互相关',corrcoef)
    dwts = TimeSeriesSimilarity(arr1, arr2)
    print('DWT',dwts)
    fdis = FrechetDistance(arr1.reshape(-1,1), arr2.reshape(-1,1))
    print('F距离',fdis)
    
    corrcoef = compute_correlaion_with_shift(arr1, arr3)[0,1]
    print('互相关',corrcoef)
    dwts = TimeSeriesSimilarity(arr1, arr3)
    print('DWT',dwts)
    fdis = FrechetDistance(arr1.reshape(-1,1), arr3.reshape(-1,1))
    print('F距离',fdis)
    
    corrcoef = compute_correlaion_with_shift(arr1, arr4)[0,1]
    print('互相关',corrcoef)
    dwts = TimeSeriesSimilarity(arr1, arr4)
    print('DWT',dwts)
    fdis = FrechetDistance(arr1.reshape(-1,1), arr4.reshape(-1,1))
    print('F距离',fdis)
    