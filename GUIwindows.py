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

# draw
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# GUI
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot #定义信号事件
from PyQt5.QtWidgets import QMessageBox # 弹出提示窗口
from PyQt5.QtWidgets import QTableWidgetItem # 展示表格所需的基本类
from PyQt5.QtWidgets import QVBoxLayout # 绘图时需要添加布局

from GUI.Ui_FaultDegreeGUI_m import Ui_FaultDiagnosis as Ui #导入窗口编辑器类

# self-defined utils
from config import cfg_GUI

from utils import set_random_seed
from utils.features import view_features_DTW
from utils.threshold import calc_thresholds
from run.tools import set_train_model, raw_signal_to_errors


#%% 定义辅助函数

def checkAndWarn(window,handle,true_fb='',false_fb='',need_true_fb=False):
    # 希望handle为真，如果假则会警告，为真则根据需要返回提示
    if not handle:
        QMessageBox.critical(window, "Warning", false_fb, QMessageBox.Ok)
        return False
    else:
        if(need_true_fb): QMessageBox.information(window, "info", true_fb)
        return True

def not_contains_chinese(path):
    return not re.search(r'[\u4e00-\u9fff]', path)

def hist_tied_to_frame(cfg, arrays, frame, is_train=False):
    n_bin = cfg.DRAW.HIST_BIN
    colors = cfg.DRAW.THRESHOLD_COLORS

    # Clear the previous canvas from the frame's layout
    if frame.layout().count() == 1:
        frame.layout().takeAt(0).widget().deleteLater()
        logger.info('Previous canvas cleared')

    figure = Figure()
    canvas = FigureCanvas(figure)

    ax = figure.add_subplot(111)
    ax.clear()  # Clear the previous plot

    if is_train:
        ax.hist(arrays, bins=n_bin, color='blue', label='normal signal')
        ax.set_title(cfg.TRAIN.NORMAL_PATH.split('/')[-1] + ' MAE distribution')
    else:
        ax.hist(arrays[0], bins=n_bin, color='blue', label='normal signal')
        ax.hist(arrays[1], bins=18, color='green', label='unknown signal', alpha=0.75)
        thresholds = calc_thresholds(arrays[0], method=cfg.FEATURE.USED_THRESHOLD)
        assert len(thresholds) <= len(colors), 'thresholds more than colors, checkout config'
        for i, (k, t) in enumerate(thresholds.items()):
            ax.axvline(x=t, linestyle='--', color=colors[i], label='threshold({})'.format(k))
        ax.axvline(x=arrays[1].mean(), linestyle='--', color='black', label='indicator')
        ax.set_title(cfg.INFERENCE.UNKWON_PATH.split('/')[-1] + ' MAE distribution')

    ax.legend()
    frame.layout().addWidget(canvas)

#%% 重载窗口类

class GUIWindow(QWidget): 
    '''
    该类用于创建GUI主页面，以及设置功能。
    该类是窗口的子类，比窗口拥有更多属性，比如self.ui这个窗口编辑器
    依赖自定义库GUI中定义的窗口Ui
    '''
    def __init__(self):
        super().__init__()
        self.editor = Ui() #实例化一个窗口编辑器
        self.editor.setupUi(self) #用这个编辑器生成布局
        self.editor.frameInTraining.setLayout(QVBoxLayout())
        self.editor.frameInPrediction.setLayout(QVBoxLayout())
        
        self.cfg = cfg_GUI
        self.model = None
        self.refence_errors = []

        logger.info("GUI window initialized")
        set_random_seed(self.cfg.SEED)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        if(self.cfg.LOG.OUTPUT_TO_FILE):
            logger.add(self.cfg.LOG.DIR + f'/{self.cfg.LOG.PREFIX}_{cur_time}.log', encoding='utf-8')
    
    @pyqtSlot() #导入正常信号 for 特征筛选
    def on_btnImportNormalSignalInSelection_clicked(self):
        fname,_ = QFileDialog.getOpenFileName(self, "导入正常信号","./", "Comma-Separated Values(*.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        logger.info("Normal signal imported: {}".format(fname))
        self.cfg.TRAIN.NORMAL_PATH = fname

    @pyqtSlot() #导入故障信号 for 特征筛选
    def on_btnImportFaultSignalInSelection_clicked(self):
        fname,_ = QFileDialog.getOpenFileName(self, "导入故障信号","./", "Comma-Separated Values(*.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        logger.info("Fault signal imported: {}".format(fname))
        self.cfg.TRAIN.FAULT_PATH = fname

    @pyqtSlot() #导入正常信号 for 模型训练
    def on_btnImportNormalSignal_clicked(self):
        fname,_ = QFileDialog.getOpenFileName(self, "导入正常信号","./", "Comma-Separated Values(*.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        if self.cfg.TRAIN.NORMAL_PATH and Path(self.cfg.TRAIN.NORMAL_PATH).exists():
            checkAndWarn(self, fname == self.cfg.TRAIN.NORMAL_PATH, 
                         false_fb="导入的正常信号与过往导入的正常信号不一致，若坚持使用不一致的数据，请无视该警告")
        logger.info("Normal signal imported: {}".format(fname))
        self.cfg.TRAIN.NORMAL_PATH = fname

    @pyqtSlot() #导入正常信号 for 新数据预测
    def on_btnImportInferentSignal_clicked(self):
        fname,_ = QFileDialog.getOpenFileName(self, "导入正常信号","./", "Comma-Separated Values(*.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        if self.cfg.TRAIN.NORMAL_PATH and Path(self.cfg.TRAIN.NORMAL_PATH).exists():
            checkAndWarn(self, fname == self.cfg.TRAIN.NORMAL_PATH, 
                         false_fb="导入的正常信号与过往导入的正常信号不一致，若坚持使用不一致的数据，请关闭该警告窗口")
        logger.info("Normal signal imported: {}".format(fname))
        self.cfg.TRAIN.NORMAL_PATH = fname

    @pyqtSlot() #导入故障信号 for 新数据预测
    def on_btnImportUnknownSignal_clicked(self):
        fname,_ = QFileDialog.getOpenFileName(self, "导入未知信号","./", "Comma-Separated Values(*.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        logger.info("Unknown signal imported: {}".format(fname))
        self.cfg.INFERENCE.UNKWON_PATH = fname

    @pyqtSlot() #导入预测模型
    def on_btnImportModel_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入预测模型","./", "PyTorch model(*.pth)")
        logger.info("Model imported: name {}, type {}".format(fname, ftype))
        if fname and not checkAndWarn(self,fname[-4:]=='.pth',false_fb="选中的文件并非.pth类型，请检查"): return
        if(fname): 
            logger.info("Model imported: {}".format(fname))
            self.model = tload(fname)
    
    @pyqtSlot() # 计算DTW并展示
    def on_btnCalculateDTW_clicked(self):
        state = self.cfg.TRAIN.NORMAL_PATH and self.cfg.TRAIN.FAULT_PATH
        if not checkAndWarn(self,state,
                            "数据导入成功，开始计算",
                            "数据缺失，请导入正常与故障信号",
                            True): return
 
        logger.info("Search propre features...")
        ranked_feat = view_features_DTW(self.cfg) # list with (feature, DTW score)
        logger.info("features ranked:\n{}".format('\n'.join(f"{k}: {v}" for k, v in ranked_feat))) 

        # 将排序后的列表转换为 Pandas DataFrame
        df = pd.DataFrame(ranked_feat, columns=["feature", "DTW score"])
        # 在 QTableWidget 中显示 DataFrame
        self.editor.tableWidget.setRowCount(len(df))
        self.editor.tableWidget.setColumnCount(len(df.columns))
        for row in range(len(df)):
            for col in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[row, col]))
                self.editor.tableWidget.setItem(row, col, item)
        # 设置列宽度为内容适应
        self.editor.tableWidget.resizeColumnsToContents()

        # 重设下拉多选框的选项
        self.editor.comboBoxSelectFeaturesInTraining.clear()
        self.editor.comboBoxSelectFeaturesInTraining.addItems([i[0] for i in ranked_feat])
        self.editor.comboBoxSelectFeaturesInTraining.selectItems([0]) # 默认选中第一个
        self.editor.comboBoxSelectFeaturesInPrediction.clear()
        self.editor.comboBoxSelectFeaturesInPrediction.addItems([i[0] for i in ranked_feat])
        self.editor.comboBoxSelectFeaturesInPrediction.selectItems([0])

    @pyqtSlot() # 训练LSTM模型
    def on_btnTraining_clicked(self):
        # 检查每个项是否被选中，如果被选中则添加到selected_items中
        items = []
        items += self.editor.comboBoxSelectFeaturesInTraining.currentData()
        # 打印已选择的选项文本
        if not checkAndWarn(self,items,false_fb="未选中任何特征"): return
        print("已选择的特征:", ", ".join(items))
        self.cfg.FEATURE.USED_F = items
        
        # 判断是否导入正常信号
        if not checkAndWarn(self, self.cfg.TRAIN.NORMAL_PATH,
                            false_fb="数据缺失，请导入正常信号"): return
        
        # 开始训练
        logger.info("Start training with config:{}".format(self.cfg))
        logger.info("feature(s) used: {}".format(', '.join(self.cfg.FEATURE.USED_F)))
        self.model = set_train_model(self.cfg)

        # 计算阈值
        logger.info('Start to calculate threshold and distribution...')
        normal_errors = raw_signal_to_errors(self.cfg, self.model, is_normal=True)
        self.refence_errors = normal_errors

        # 绘制并显示
        hist_tied_to_frame(self.cfg,normal_errors,self.editor.frameInTraining,is_train=True)
 
    @pyqtSlot() # 保存LSTM模型
    def on_btnSaveModel_clicked(self):
        if not checkAndWarn(self,self.model,false_fb="模型不存在，请训练模型"): return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存模型", "LSTM", "Model Files (*.pth)")
        if file_path:
            # 检查路径是否存在中文
            if not checkAndWarn(self,not_contains_chinese(file_path),
                                false_fb="路径含有中文，无法保存",
                                true_fb=f"模型已保存到：{file_path}",
                                need_true_fb=True): return
            tsave(self.model, file_path)
            
    @pyqtSlot() # 对新数据进行预测
    def on_btnPredict_clicked(self):
        logger.info("Start checkout config matching...")
        # 检测是否有未知数据
        if not checkAndWarn(self,self.cfg.INFERENCE.UNKWON_PATH,false_fb="请导入待测信号"): return
        # 检测是否有正常信号
        if not checkAndWarn(self,self.cfg.TRAIN.NORMAL_PATH,false_fb="请导入作为参考的正常信号"): return
        # 检测是否有模型
        if not checkAndWarn(self,self.model,false_fb="请导入预测模型"): return
        # 检测是否选择特征
        pointed_features = self.editor.comboBoxSelectFeaturesInPrediction.currentData()
        if pointed_features: self.cfg.FEATURE.USED_F = pointed_features
        if not checkAndWarn(self,self.cfg.FEATURE.USED_F,false_fb="未选中任何特征"): return
        # 检查是否已经计算了正常信号的MAE，没有计算则补上
        if not self.refence_errors.any():
            logger.info('Start to calculate normal signal MAE...')
            self.refence_errors = raw_signal_to_errors(self.cfg, self.model, is_normal=True)
        
        # 读取模型的输入维数并检查
        input_dim = self.model.lstm.input_size #模型的输入维数/通道数
        data_tunnel = pd.read_csv(self.cfg.INFERENCE.UNKWON_PATH).shape[1] #数据表列数
        feature_n = len(self.cfg.FEATURE.USED_F)
        state = (feature_n * data_tunnel == input_dim)
        if not checkAndWarn(self,state,false_fb=f'模型输入维数{input_dim}与数据通道数{data_tunnel}、特征数{feature_n}不匹配'): return

        # 计算未知信号MAE
        logger.info('Start to calculate unknown signal MAE...')
        errors = raw_signal_to_errors(self.cfg, self.model, is_normal=False)
        hist_tied_to_frame(self.cfg, [self.refence_errors,errors], 
                           self.editor.frameInPrediction ,is_train=False)
        

#%% 开始运行

app = QApplication(sys.argv)
# app.setQuitOnLastWindowClosed(True) #添加这行才能在spyder正常退出

w = GUIWindow()
w.show()

n = app.exec()
sys.exit(n)