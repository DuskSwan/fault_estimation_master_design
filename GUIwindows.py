# -*- coding: utf-8 -*-

# utils
import sys
import time
import re
from pathlib import Path
from loguru import logger

# data
import pandas as pd
import torch.nn as nn
from torch import save as tsave
from torch import load as tload

# GUI
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot #定义信号事件

from PyQt5.QtWidgets import QMessageBox # 弹出提示窗口
from PyQt5.QtWidgets import QTableWidgetItem # 展示表格所需的基本类

from GUI.FaultDegreeGUI import Ui_Form #导入窗口编辑器类

# self-defined utils

from config import cfg

from utils import set_random_seed,initiate_cfg
from utils.features import view_features_DTW

from run.tools import signal_to_XY

from data import make_data_loader
from modeling import build_model
from solver import make_optimizer

from engine.trainer import do_train

#%% 定义辅助函数

def checkAndWarn(window,handle,true_fb='',false_fb='',need_true_fb=False):
    # 希望handle为真，如果假则会警告
    if not handle:
        QMessageBox.critical(window, "Warning",false_fb, QMessageBox.Ok)
        return False
    else:
        if(need_true_fb): QMessageBox.information(window, "info", true_fb)
        return True

def calcMAEWithLSTM(path,lstm,used_func,
                    sublen,piece,m,p,fpiece,
                    name=''):
    data = pd.read_csv(path).values #numpy数组
    # data = data[:1200000] #这里是减少所用的信号长度
    
    XY = DF.sheet_cut(data, sublen, piece, method = 1, show_para = False)
    f_df = SF.signal_to_features_tf(XY,feat_func_name_list = used_func) #提取特征
    
    score = SF.test_features_predict(f_df,m,p,lstm,piece=fpiece,
                                  need_plot=False, title=name,
                                  view_sample_idx = 0) #预测查看误差
    s = score.mean() #计算平均误差
    print('{},mean MAE={}'.format(name,s))
    return s

def not_contains_chinese(path):
    return not re.search(r'[\u4e00-\u9fff]', path)

def calc_threshold(normal_idx,fault_idx, type = 'linear', lmd = 0.2):
    assert fault_idx > normal_idx, 'fault_idx should be larger than normal_idx'
    if type == 'linear': return (fault_idx - normal_idx) * lmd + normal_idx
    elif type == 'exp': return exp(log(fault_idx - normal_idx) * lmd) + normal_idx
    else: return normal_idx * (1 + lmd)


#%% 重载窗口类

class GUIWindow(QWidget): 
    '''
    该类用于创建GUI主页面，以及设置功能。
    该类是窗口的子类，比窗口拥有更多属性，比如self.ui这个窗口编辑器
    依赖自定义库GUI_view中的类Ui_Form
    '''
    def __init__(self):
        super().__init__()
        self.editor = Ui_Form() #实例化一个窗口编辑器
        self.editor.setupUi(self) #用这个编辑器生成布局
        
        self.sublen_of_draw_features = 2048
        self.m = 50
        self.p = 5
        self.f_piece = 100 #需要的特征样本数
        self.piece = self.f_piece * (self.m+self.p) # 需要的原信号样本数
            #为了满足lstm的训练要求，最开始必须划分出足够多的样本
        self.lstm_epoch = 100
        self.lstm_batch_size = 32
        
        self.normalSignalForSelectionPath = None
        self.faultSignalForSelectionPath = None
        self.normalSignalForTrainingPath = None
        self.faultSignalForTrainingPath = None
        self.signalForPredctionPath = None
            # 要求是无列名的csv数据表
        self.seletedFeatureNames = None
        self.threshold = 0
        self.normal_MAE = 0
        self.fault_MAE = 1e10
        self.lstm = None
        self.seletedFeatureNames = []
    
    @pyqtSlot() #导入正常信号 for 特征筛选
    def on_btnImportNormalSignalInSelection_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入正常信号","./", "All Files(*);;Comma-Separated Values(.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        self.normalSignalForSelectionPath = fname     
    @pyqtSlot() #导入故障信号 for 特征筛选
    def on_btnImportFaultSignalInSelection_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入故障信号","./", "All Files(*);;Comma-Separated Values(.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        self.faultSignalForSelectionPath = fname
    @pyqtSlot() #导入正常信号 for 模型训练
    def on_btnImportNormalSignalInTraining_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入正常信号","./", "All Files(*);;Comma-Separated Values(.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        self.normalSignalForTrainingPath = fname        
    @pyqtSlot() #导入故障信号 for 模型训练
    def on_btnImportFaultSignalInTraining_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入故障信号","./", "All Files(*);;Comma-Separated Values(.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        self.faultSignalForTrainingPath = fname
    @pyqtSlot() #导入故障信号 for 新数据预测
    def on_btnImportSignalInPrediction_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入未知信号","./", "All Files(*);;Comma-Separated Values(.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        self.signalForPredctionPath = fname
    @pyqtSlot() #导入预测模型
    def on_btnImportModel_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入预测模型","./", "All Files(*);;PyTorch model(.pt)")
        if fname and not checkAndWarn(self,fname[-3:]=='.pt',false_fb="选中的文件并非.pt类型，请检查"): return
        if(fname): self.lstm = tload(fname)
    
    @pyqtSlot() # 计算DTW并展示
    def on_btnCalculateDTW_clicked(self):
        state = self.normalSignalForSelectionPath and self.faultSignalForSelectionPath
        if not checkAndWarn(self,state,
                            "数据导入成功，开始计算",
                            "数据缺失，请导入正常与故障信号",
                            True): return
        # 计算特征
        tpaths = [self.normalSignalForSelectionPath,self.faultSignalForSelectionPath]      
        feat_with_classes = [] # 每个元素是一个类别的特征序列矩阵
        for i in range(len(tpaths)): #每个类别
            data_path = tpaths[i]
            data = pd.read_csv(data_path).values #numpy数组
            data = data[:1024000] #这里是减少所用的信号长度
            XY = DF.sheet_cut(data, self.sublen_of_draw_features, method = 0, show_para = False)
            # XY = DF.sheet_cut(data, self.sublen_of_draw_features, view_piece, method = 1, show_para = False)
            f_df = SF.signal_to_features_tf(XY, output_type='pd') #提取特征
            feat_with_classes.append(f_df)
        # 对特征排序
        cols = feat_with_classes[0].columns
        feat_mark = {}
        for col in cols:
            arr1 = feat_with_classes[0][col]
            arr2 = feat_with_classes[1][col]
            dtws = TF.TimeSeriesSimilarity(arr1, arr2)
            txt = '{:20}: dtw= {:.6f} '.format(col,dtws)
            print(txt)
            feat_mark[col] = dtws
        d_order=sorted(feat_mark.items(),key=lambda x:x[1])[::-1]
        # for i in d_order: print(i)
        # 将排序后的列表转换为 Pandas DataFrame
        df = pd.DataFrame(d_order, columns=["feature", "DTW score"])
        # 在 QTableWidget 中显示 DataFrame
        self.editor.tableWidget.setRowCount(len(df))
        self.editor.tableWidget.setColumnCount(len(df.columns))
        for row in range(len(df)):
            for col in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[row, col]))
                self.editor.tableWidget.setItem(row, col, item)
        # 设置列宽度为内容适应
        self.editor.tableWidget.resizeColumnsToContents()

    @pyqtSlot() # 训练LSTM模型
    def on_btnTraining_clicked(self):
        # 检查每个项是否被选中，如果被选中则添加到selected_items中
        items = []
        items += self.editor.comboBoxSelectFeaturesInTraining.currentData()
        # 打印已选择的选项文本
        if not checkAndWarn(self,items,false_fb="未选中任何特征"): return
        print("已选择的特征:", ", ".join(items))
        self.seletedFeatureNames = items
        
        # 判断是否导入
        state = self.normalSignalForTrainingPath and self.faultSignalForTrainingPath
        if not checkAndWarn(self,state,false_fb="数据缺失，请导入正常与故障信号"): return
        
        # 设置参数
        normal_data = pd.read_csv(self.normalSignalForTrainingPath).values #读成numpy数组
        n,_ = normal_data.shape # 1958912
        piece = self.piece
        subl = self.sublen_of_draw_features
        
        # 判断采样点数是否足够
        state = n > piece + subl
        if not checkAndWarn(self,state,false_fb=f"原始采样点数为{n}，所需采样点数为{piece}+{subl}={piece+subl}，无法达标"): return
        
        # 特征提取
        XY = DF.sheet_cut(normal_data, subl, piece, method = 3, show_para = True)
            # 这一步需要保证特征是连续提取的
        st = time.time()
        normal_f = SF.signal_to_features_tf(XY,feat_func_name_list = self.seletedFeatureNames) 
            # 形状(piece,p_feature)
        et = time.time()
        print(normal_f.shape)
        print('draw time:', et-st)
        
        # 训练LSTM
        more_para = {'lstm_hidden_dim':10}
        self.lstm = SF.train_lstm_with_features(normal_f, m=self.m, p=self.p, piece=self.f_piece,
                                           need_view_loss = False, 
                                           epochs = self.lstm_epoch, 
                                           bs = self.lstm_batch_size,
                                           **more_para)
        
        # 计算MAE
        params = {'m':self.m, 'p':self.p,
                  'piece':self.piece, 'fpiece':self.f_piece,
                  'lstm':self.lstm, 'used_func':self.seletedFeatureNames,
                  'sublen':self.sublen_of_draw_features}
        normal_MAE = calcMAEWithLSTM(self.normalSignalForTrainingPath,name='normal signal',**params)
        fault_MAE = calcMAEWithLSTM(self.faultSignalForTrainingPath,name='fault signal',**params)
        
        # 展示MAE
        self.editor.lineEditNormalMAE.setText(str(normal_MAE))
        self.editor.lineEditFaultMAE.setText(str(fault_MAE))
        
        # 记录阈值、误差上下限
        self.threshold = calc_threshold(normal_MAE,fault_MAE, 'exp', 0.2)
        self.normal_MAE = normal_MAE
        self.fault_MAE = fault_MAE
        self.editor.lineEditMAEThreshold.setText(str(self.threshold))
        
    @pyqtSlot() # 保存LSTM模型
    def on_btnSaveModel_clicked(self):
        if not checkAndWarn(self,self.lstm,false_fb="模型不存在，请训练模型"): return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存模型", "LSTM", "Model Files (*.pt);;All Files (*)")
        if file_path:
            # 检查路径是否存在中文
            if not checkAndWarn(self,not_contains_chinese(file_path),
                                false_fb="路径含有中文，无法保存",
                                true_fb=f"模型将保存到：{file_path}",
                                need_true_fb=True): return
            tsave(self.lstm, file_path)
            
    @pyqtSlot() # 对新数据进行预测
    def on_btnPredict_clicked(self):
        # 检测是否有未知数据
        if not checkAndWarn(self,self.signalForPredctionPath,false_fb="请导入待测信号"): return
        # 检测是否有模型
        if not checkAndWarn(self,self.lstm,false_fb="请导入预测模型"): return
        # 检测是否选择特征，未选择就用训练时指定的
        pointed_features = self.editor.comboBoxSelectFeaturesInPrediction.currentData()
        if pointed_features: self.seletedFeatureNames = pointed_features
        if not checkAndWarn(self,self.seletedFeatureNames,false_fb="未选中任何特征"): return
        # 检测是否设定阈值
        pointed_threshold_text = self.editor.lineEditMAEThreshold.text()
        if not checkAndWarn(self,pointed_threshold_text,false_fb="未设定阈值"): return
        self.threshold = float(pointed_threshold_text)
        
        # 读取模型的输入维数并检查
        input_dim = self.lstm.lstm.input_size #模型的输入维数/通道数
        data_tunnel = pd.read_csv(self.signalForPredctionPath).shape[1] #数据表列数
        feature_n = len(self.seletedFeatureNames)
        state = (feature_n * data_tunnel == input_dim)
        if not checkAndWarn(self,state,false_fb=f'模型输入维数{input_dim}与数据通道数{data_tunnel}、特征数{feature_n}不匹配'): return
    
        # 进行预测
        params = {'m':self.m, 'p':self.p,
                  'piece':self.piece, 'fpiece':self.f_piece,
                  'lstm':self.lstm, 'used_func':self.seletedFeatureNames,
                  'sublen':self.sublen_of_draw_features}
        new_MAE = calcMAEWithLSTM(self.signalForPredctionPath,name='unknown signal',**params)
        is_fault = (new_MAE > self.threshold)
        
        # 展示结果
        self.editor.lineEditPredictionMAE.setText(str(new_MAE))
        if is_fault: self.editor.lineEditResult.setText('该信号故障')
        else: self.editor.lineEditResult.setText('该信号正常')
        

#%% 开始运行

app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(True) #添加这行才能在spyder正常退出

w = GUIWindow()
w.show()

n = app.exec()
sys.exit(n)