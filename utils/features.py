import pandas as pd
import numpy as np

from . import sheet_cut
from .similarity import TimeSeriesSimilarity

# 时域/统计量
def origin(signal): #原始值
    return signal[-1]
def Max(signal): # 
    # try: return np.max(signal)
    # except: print(signal)
    return np.nanmax(signal) 
def Min(signal): # 
    return np.nanmin(signal) 
def Ms(signal): #均方值
    return np.mean(signal**2) 
def Mean(signal): # 均值 t1
    return np.mean(signal) 
def Sra(signal): #幅值平方根 t2
    t=np.sqrt(np.abs(signal))
    return np.mean(t)**2
def Rms(signal): # 均方根 t3
    return np.sqrt(np.mean(signal**2)) 
def t4(signal): # 幅值均值 t4
    return np.mean(np.abs(signal))
def t5(signal): # 3阶原点矩 t5
    return np.mean(signal**3)
def t6(signal): # 4阶原点矩 t6
    return np.mean(signal**4)
def Ppk(signal): # 峰峰值/极差 t7
    return np.max(signal)-np.min(signal)
def t8(signal): #幅值峰值/幅值极大值 t8
    return np.max(np.abs(signal))
def t9(signal): #极小值 t9
    return np.min(signal)
def Var(signal): #样本方差 t10
    return np.var(signal, ddof=1) 
def Std(signal): # 样本标准差 t11
    return np.std(signal, ddof=1) 
def t12(signal): # t12
    t3 = np.sqrt(np.mean(signal**2))
    t1 = np.mean(signal) 
    return t3/t1
def ShapeFactor(signal): # 形状因子 t13
    m=np.max(np.abs(signal))
    s=np.mean(signal**2) ** 0.5
    return m/s**0.5
def t14(signal): #t14
    return np.max(signal)/np.abs(np.mean(signal))
def Kurtosis(signal): # 峭度/峰度 t15
    N=len(signal)
    s=np.std(signal)
    t=(signal-np.mean(signal))**4
    return np.sum(t)/(N * s**4)
def Skewness(signal): # 偏斜度/偏态 t16
    N=len(signal)
    s=np.std(signal)
    t=(signal-np.mean(signal))**3
    return np.sum(t)/(N * s**3)
def KurtosisFactor(signal): # 波峰因子/峰值因子 t17
    return Kurtosis(signal)/Ms(signal)**2
def ImpulseFactor(signal): # 脉冲因子 t18
    m=np.max(np.abs(signal))
    s=np.mean(np.abs(signal))
    return m/s
def Cf(signal): #峰值系数 t19
    m = np.max(np.abs(signal))
    s = np.mean(signal**2) ** 0.5
    return m/s
def Mf(signal): #裕度因子 t20
    m=np.max(np.abs(signal))
    s=np.mean(np.abs(signal)) ** 2
    return m/s
def KurtosisFactor(signal): #峭度因子
    m=Kurtosis(signal)
    s=np.mean(signal**2) ** 2
    return m/s

# 定义一些同名函数
RMS = Rms
SRA = Sra
KV = Kurtosis
SV = Skewness
PPV = Ppk
CF = CrestFactor = Cf
IF = ImpulseFactor
MF = Mf
SF = ShapeFactor
KF = KurtosisFactor

# 频域
def draw_fi(y): #提取出频率谱向量
    #依赖numpy库,以np引用
    #输入是一段信号，长度N
    #输出是实数频率向量，长度N//2
    N=len(y)
    fft_y=np.fft.fft(y) #快速傅里叶变换
    abs_y=np.abs(fft_y) #取复数的绝对值，即频率幅值
    f=abs_y[:N//2] #取前一半
    return f 
def fMean(signal,f=None): #中心频率/频率均值 p1
    if(f is None): f = draw_fi(signal)
    return np.mean(f)
def Rmsf(signal,f=None): #RMS频率/频率均方根
    if(f is None): f = draw_fi(signal)
    return np.mean(f**2)**0.5
def Rvf(signal,f=None): #根方差频率/频率标准差
    if(f is None): f = draw_fi(signal)
    df=f-np.mean(f)
    return np.mean(df**2)**0.5
def p2(signal,f=None): #谱方差
    if(f is None): f = draw_fi(signal)
    N=len(f)
    s=np.sum((f-np.mean(f))**2)
    return s/(N-1)
def p3(signal,f=None): #三阶矩
    if(f is None): f = draw_fi(signal)
    N=len(f)
    s=np.sum((f-np.mean(f))**3)
    return s/(N * np.std(signal)**3)
def p4(signal,f=None): #四阶矩
    if(f is None): f = draw_fi(signal)
    N=len(f)
    s=np.sum((f-np.mean(f))**4)
    return s/(N * np.std(signal)**4)
def p5(signal,f=None): 
    if(f is None): f = draw_fi(signal)
    N=len(f)
    nt=0
    for k in range(N):
        nt+=k*f[k]
    return nt/np.mean(f)*N
def p6(signal,f=None):
    if(f is None): f = draw_fi(signal)
    N=len(f)
    temp=p5(signal)
    nt=0
    for k in range(N):
        nt += (k-temp)**2 * f[k]
    return np.sqrt(nt/N)
def p7(signal,f=None):
    if(f is None): f = draw_fi(signal)
    N=len(f)
    nt=0
    for k in range(N):
        nt+=k*k*f[k]
    return np.sqrt(nt/np.mean(f))
def p8(signal,f=None):
    if(f is None): f = draw_fi(signal)
    N=len(f)
    nt=0
    dnt=0
    for k in range(N):
        nt += k**4 *f[k]
        dnt += k**2 *f[k]
    return np.sqrt(nt/dnt)
def p9(signal,f=None):
    if(f is None): f = draw_fi(signal)
    N=len(f)
    nt=0
    dnt=0
    for k in range(N):
        nt += k**4 *f[k]
        dnt += k**2 *f[k]
    return np.sqrt(nt/dnt)
def p10(signal,f=None):
    return p6(signal,f)/p5(signal,f)
def p11(signal,f=None):
    if(f is None): f = draw_fi(signal)
    N=len(f)
    fi=np.arange(N)
    s = (fi-p5(signal,f))**3 * f
    return np.sum(s)/(N * p6(signal,f)**3)
def p12(signal,f=None): 
    if(f is None): f = draw_fi(signal)
    N=len(f)
    fi=np.arange(N)
    s = (fi-p5(signal,f))**1 * f 
        #此处的幂指数1在老师给的公式中是0.5，疑为错误。因为底数并非恒正
    return np.sum(s)/(N * p6(signal,f))

FC = fMean
RMSF = Rmsf
RVF = Rvf

# 功能函数

def view_features_DTW(cfg):
    # 计算特征
    tpaths = [cfg.DATASETS.NORMAL_PATH, cfg.DATASETS.FAULT_PATH]
    feat_with_classes = [] # 每个元素是一个类别的特征序列矩阵
    for i in range(len(tpaths)): #每个类别
        data_path = tpaths[i]
        data = pd.read_csv(data_path).values #numpy数组
        n,_ = data.shape
        length = min(n, cfg.FEATURE.MAX_LENGTH)
        data = data[:length] #减少所用的信号长度
        XY = sheet_cut(data, cfg.DESIGN.SUBLEN, method = 0)
        f_df = signal_to_features_tf(XY, output_type='pd') #提取特征
        feat_with_classes.append(f_df)
    # 对特征排序
    cols = feat_with_classes[0].columns
    feat_mark = {}
    for col in cols:
        arr1 = feat_with_classes[0][col]
        arr2 = feat_with_classes[1][col]
        dtws = TimeSeriesSimilarity(arr1, arr2)
        # txt = '{:20}: dtw= {:.6f} '.format(col,dtws)
        # print(txt)
        feat_mark[col] = dtws
    ranked_feat=sorted(feat_mark.items(),key=lambda x:x[1])[::-1] # 是列表，每个元素为(特征，DTW得分)
    for k,v in ranked_feat: print('{:20}: dtw= {:.6f} '.format(k,v))
    return ranked_feat

def signal_to_features_tf(sample_list, output_type='np',feat_func_name_list = None):
    """
    sample_list是时间序列切片得到的样本集，是三维数组（单个样本是二维数组），每个样本计算一个向量作为特征
    output_type指定输出类型的数组（np）还是数据表（pd）
    输出特征组成的时间序列，是二维数组
    """
    feature_df = pd.DataFrame([])
    _,_,n_tun = sample_list.shape #读取通道数
    for sample in sample_list: #整理特成为表格
        feature_dict = {}
        for i in range(n_tun):
            tun_signal = sample[:,i] #选第i通道
            i_dict = extract_features(tun_signal, 'tun'+str(i), feat_func_name_list)
            feature_dict.update(i_dict)
        # feature_df = feature_df.append(feature_dict, ignore_index=True)
            # ==warning:The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
        df_dictionary = pd.DataFrame([feature_dict])
        feature_df = pd.concat([feature_df, df_dictionary], ignore_index=True)
    # print(feature_df.columns)
    # feature_df['class'] = y #加入标签一列
    
    if(output_type=='pd'): return feature_df
    elif(output_type=='np'): return feature_df.values
    else: raise ValueError('unknown output_type '+output_type)

def extract_features(signal, prefix='', feat_func_name_list = None):
    '''
    该函数针对一段信号，提取出全部的特征，返回特征构成的(特征名，特征值)字典
    prefix是保存名字时的前缀
    feat_func_name_list是特征函数名字的列表，可以指定特征函数
    '''
    
    if(feat_func_name_list is None):
        # 不指定特征时，使用以下默认特征
        funcs = [RMS, SRA, KV, SV, PPV,
                 CF, IF, MF, SF, KF,
                 FC, RMSF, RVF,
                 Mean, Var, Std, Max, Min,
            ]
    else:
        assert isinstance(feat_func_name_list, list), 'feat_func_name_list has to be list.'
        assert isinstance(feat_func_name_list[0], str), 'elements in feat_func_name_list has to be string.'
        funcs = []
        for name in feat_func_name_list:
            try: funcs.append(eval(name))
            except: raise ValueError(f"not find function {name}")

    features={} #保存特征
    for i in range(len(funcs)):
        f = funcs[i]
        name = prefix + '_' + f.__name__
        value = f(signal)
        features[name] = value
    return features