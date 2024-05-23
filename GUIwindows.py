# -*- coding: utf-8 -*-

#%% import

# utils
import sys
import time
import re
from pathlib import Path
from loguru import logger

# data
import numpy as np
import pandas as pd
# import torch.nn as nn
from torch import save as tsave
from torch import load as tload

# draw
import tempfile
# import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import DateFormatter, AutoDateLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# GUI
from PyQt5.QtWidgets import QApplication, QWidget # 导入窗口类
from PyQt5.QtWidgets import QFileDialog # 导入文件对话框
from PyQt5.QtWidgets import QMessageBox # 弹出提示窗口
from PyQt5.QtWidgets import QAction # 导入菜单栏
from PyQt5.QtWidgets import QTableWidgetItem # 展示表格所需的基本类
from PyQt5.QtWidgets import QVBoxLayout # 绘图时需要添加布局
# from PyQt5.QtWidgets import QDialog, QGridLayout # 在新窗口中放大显示图形
# from PyQt5.QtWidgets import QMenuBar # 菜单栏
from PyQt5.QtWidgets import QMainWindow # 主窗口
from PyQt5.QtCore import pyqtSlot, pyqtSignal # 定义信号事件
from PyQt5.QtCore import QThread # 定义线程
from PyQt5.QtCore import QEvent, Qt # 检测事件和键盘按键

# from GUI.Ui_FaultDegreeGUI_m import Ui_FaultDiagnosis as Ui # 导入窗口编辑器类
from GUI.Ui_MainWindow_m import Ui_MainWindow as Ui # 导入窗口编辑器类
from GUI.OpenPlotFrame import EnlargedWindow # 导入放大显示图形的窗口类
from GUI.SettingMenu import SetParametersDialog # 导入设置参数的窗口类

# self-defined utils
from config import cfg_GUI

from utils import set_random_seed, sort_list
from utils.features import view_features_DTW_with_n_normal
from utils.threshold import calc_thresholds
from utils.denoise import array_denoise
from run.tools import set_train_model, raw_signal_to_errors


#%% 定义辅助函数

def checkAndWarn(window,handle,true_fb='',false_fb='',need_true_fb=False):
    # 希望handle为真，如果假则会警告，为真则根据需要返回提示
    # 返回handle的真值
    if not handle:
        QMessageBox.critical(window, "Warning", false_fb, QMessageBox.Ok)
        return False
    else:
        if(need_true_fb): QMessageBox.information(window, "info", true_fb)
        return True

def not_contains_chinese(path):
    return not re.search(r'[\u4e00-\u9fff]', path)

def hist_tied_to_frame(cfg, arrays, canvas, is_train=False):
    n_bin = cfg.DRAW.HIST_BIN
    colors = cfg.DRAW.THRESHOLD_COLORS

    # 清理之前的图形
    figure = canvas.figure
    figure.clear()
    ax = figure.add_subplot(111)
    ax.clear()  # Clear the previous plot

    if is_train:
        ax.hist(arrays, bins=n_bin, color='blue', label='normal signal')
        name = Path(cfg.TRAIN.NORMAL_PATH).stem
        ax.set_title(name + ' MAE distribution')
    else:
        ax.hist(arrays[0], bins=n_bin, color='blue', label='normal signal')
        ax.hist(arrays[1], bins=18, color='green', label='unknown signal', alpha=0.75)
        thresholds = calc_thresholds(arrays[0], method=cfg.FEATURE.USED_THRESHOLD)
        assert len(thresholds) <= len(colors), 'thresholds more than colors, checkout config'
        for i, (k, t) in enumerate(thresholds.items()):
            ax.axvline(x=t, linestyle='--', color=colors[i], label='threshold({})'.format(k))
        ax.axvline(x=arrays[1].mean(), linestyle='--', color='black', label='indicator')
        name = Path(cfg.INFERENCE.UNKWON_PATH).stem
        ax.set_title(name + ' MAE distribution')

    ax.legend()
    # 强制刷新画布
    figure.tight_layout()
    canvas.draw()

def update_ratio_to_frame(cfg, series: pd.Series, canvas, tit=''):
    logger.info('Plot time series \n {}'.format(series))    

    # 清理之前的图形
    figure = canvas.figure
    figure.clear()
    ax = figure.add_subplot(111)
    ax.clear()  # Clear the previous plot

    # 尝试检测索引类型（时间戳或整数）
    try:
        # 尝试将索引转换为DatetimeIndex
        temp_index = pd.to_datetime(series.index, format='%Y.%m.%d.%H.%M.%S', errors='raise')
        # 如果成功，假设是时间戳
        is_timestamp = True
    except ValueError:
        # 转换失败，假设是整数
        is_timestamp = False

    if is_timestamp:
        # 使用日期格式化
        x = temp_index
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(AutoDateLocator())
    else:
        # 索引被假设为整数，直接使用
        x = pd.RangeIndex(start=0, stop=len(series))
    
    # cont = Path(cfg.INFERENCE.TEST_CONTENT)
    y = series.values
    interval = len(x) // 10 if len(x) > 30 else 1
    xticks = x[::interval] if not is_timestamp else x[::interval].strftime("%Y-%m-%d")

    if(cfg.DENOISE.SHOW_TYPE == 'original only'):
        ax.plot(x, y)
        idx = np.argmax(y > cfg.INFERENCE.MAE_ratio_threshold)
        if(idx): # 如果有大于阈值的元素
            ax.scatter(x[idx], y[idx], c='r', label='MAE ratio > threshold')
            ax.text(x[idx], y[idx], f'({x[idx]})', ha='right', color='r')
    elif(cfg.DENOISE.SHOW_TYPE == 'denoised only'):
        denoised_y = array_denoise(y, 
                                   method=cfg.DENOISE.SHOW_METHOD, 
                                   step=cfg.DENOISE.SHOW_SMOOTH_STEP, 
                                   wavelet=cfg.DENOISE.SHOW_WAVELET, 
                                   level=cfg.DENOISE.SHOW_LEVEL)
        denoised_y = np.clip(denoised_y, 0, 1)
        ax.plot(x, denoised_y)
        idx = np.argmax(denoised_y > cfg.INFERENCE.MAE_ratio_threshold)
        ax.scatter(x[idx], denoised_y[idx], c='r', label='MAE ratio > threshold')
        ax.text(x[idx], denoised_y[idx], f'({x[idx]})', color='r',
                 ha='right')
    elif(cfg.DENOISE.SHOW_TYPE == 'both'):
        ax.plot(x, y, label='original')
        denoised_y = array_denoise(y, 
                                   method=cfg.DENOISE.SHOW_METHOD, 
                                   step=cfg.DENOISE.SHOW_SMOOTH_STEP, 
                                   wavelet=cfg.DENOISE.SHOW_WAVELET, 
                                   level=cfg.DENOISE.SHOW_LEVEL)
        denoised_y = np.clip(denoised_y, 0, 1)
        ax.plot(x, denoised_y, label='denoised')

        idx1 = np.argmax(y > cfg.INFERENCE.MAE_ratio_threshold)
        ax.scatter(x[idx1], y[idx1], label='original ratio > threshold')
        ax.text(x[idx1], y[idx1], f'({x[idx1]})', color='r',
                 ha='right')

        idx2 = np.argmax(denoised_y > cfg.INFERENCE.MAE_ratio_threshold)
        ax.scatter(x[idx2], denoised_y[idx2], label='denoised MAE ratio > threshold')
        ax.text(x[idx2], denoised_y[idx2], f'({x[idx2]})', color='r',
                 ha='right')
    else:
        raise ValueError('cfg.DENOISE.SHOW_TYPE should be one of "original only", "denoised only", "both"')
    ax.axhline(y=cfg.INFERENCE.MAE_ratio_threshold, linestyle='--', color='r', label='threshold')

    # set labels
    ax.set(ylim=(-0.1, 1.15),
           xlabel = 'Time' if is_timestamp else 'Index',
           ylabel='y',
           title = tit + ' ratio of elements greater than threshold'
           )
 
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend()

    # 强制刷新画布
    figure.tight_layout()
    canvas.draw()

def draw_heatmap(corr, canvas):
    '''
    Draw a correlation heatmap of the given corr matrix on the specified canvas.
    '''
    # 清理之前的图形
    figure = canvas.figure
    figure.clear()

    # 在现有 figure 中添加一个新的子图
    ax = figure.add_subplot(111)

    # 根据相关性矩阵并绘制热力图
    cax = ax.matshow(corr, cmap='coolwarm')
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)    
    figure.colorbar(cax, cax=cbar_ax) # 在指定的 Axes 上创建 color bar
    # figure.colorbar(cax)

    # 设置轴标签
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title('Correlation Heatmap of Features')

    # 设置x轴ticks出现在图片下方
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    # 强制刷新画布
    figure.tight_layout()
    canvas.draw()

def show_table_in_widget(df, widget):
    # 在 QTableWidget 中显示 DataFrame
    n = len(df)
    m = len(df.columns)
    widget.setRowCount(n)
    widget.setColumnCount(m)
    widget.setHorizontalHeaderLabels(df.columns)  # 设置列名
    for row in range(n):
        for col in range(m):
            value = df.iloc[row, col]
            if isinstance(value, float):
                item = QTableWidgetItem(f"{value:.6f}")
            else:
                item = QTableWidgetItem(str(value))
            widget.setItem(row, col, item)
    # 设置列宽度为内容适应
    widget.resizeColumnsToContents()

#%% 定义复杂功能的线程

class DetectThread(QThread):
    signal_turn = pyqtSignal(dict)
    def __init__(self, cfg, model, refe_errs):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.refence_errors = refe_errs
        self._is_running = True
    def run(self):   #固定函数，不可变，线程开始自动执行run函数

        # 检查是否已经计算了正常信号的MAE，没有计算则补上
        if not self.refence_errors.any():
            logger.info('Start to calculate normal signal MAE...')
            self.refence_errors = raw_signal_to_errors(self.cfg, self.model, is_normal=True)

        # 读取模型的输入维数并检查
        cont = Path(self.cfg.INFERENCE.TEST_CONTENT)
        logger.info('Full roll test data directory: {}'.format(cont))
        files = sort_list(list(cont.glob('*.csv')))
        
        # 计算MAE比较阈值
        logger.info('Start to calculate threshold...')
        thresholds = calc_thresholds(self.refence_errors, method = self.cfg.FEATURE.USED_THRESHOLD)
        threshold = thresholds['Z']

        # 全寿命数据检测
        res = {}
        for file in files:
            if not self._is_running: 
                logger.info('Detection stopped')
                return
            logger.info('Current file index: {}'.format(file.stem))
            unknown_errors = raw_signal_to_errors(self.cfg, self.model, is_normal=False, file_path=file)
            logger.info('Unkwon signal: Max error {:.4f} , Min error {:.4f}, Mean error {:.4f}'
                        .format(unknown_errors.max().item(), 
                                unknown_errors.min().item(), 
                                unknown_errors.mean().item()))

            num_greater_than_threshold = (unknown_errors > threshold).sum()
            ratio = num_greater_than_threshold / unknown_errors.size
            res[file.stem] = ratio
            logger.info(f"大于阈值的元素比例：{ratio}")
            self.signal_turn.emit(res)
        logger.info('ratio dict: {}'.format(res))
        logger.info('Detection finished')

    def stop(self):
        self._is_running = False  # 提供一个方法来改变运行状态

class PredictThread(QThread):
    signal_finish = pyqtSignal(list)
    def __init__(self, cfg, model, refe_errs):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.refence_errors = refe_errs
    def run(self):
        # 检查是否已经计算了正常信号的MAE，没有计算则补上
        if not self.refence_errors.any():
            logger.info('Start to calculate normal signal MAE...')
            self.refence_errors = raw_signal_to_errors(self.cfg, self.model, is_normal=True)
        
        # 计算未知信号MAE
        logger.info('Start to calculate unknown signal MAE...')
        errors = raw_signal_to_errors(self.cfg, self.model, is_normal=False)
        self.signal_finish.emit([self.refence_errors, errors])


#%% 重载窗口类

class GUIWindow(QMainWindow): 
    '''
    该类用于创建GUI主页面，以及设置功能。
    该类是窗口的子类，比窗口拥有更多属性，比如self.ui这个窗口编辑器
    依赖自定义库GUI中定义的窗口Ui
    '''
    def __init__(self):
        super().__init__()
        self.editor = Ui() #实例化一个窗口编辑器
        self.editor.setupUi(self) #用这个编辑器生成布局

        self.editor.frameInFeatures.setLayout(QVBoxLayout())
            # 添加一个layout，之后才能用frame.layout().addWidget(canvas)来加入画布
        self.canvasInFeatures = FigureCanvas(Figure())
        self.editor.frameInFeatures.layout().addWidget(self.canvasInFeatures)
            # 添加一个画布到layout中，这里记下来这个画布的名字以便后续更新
        self.canvasInFeatures.installEventFilter(self)
            # 安装事件过滤器到 FigureCanvas 上以便检测鼠标双击事件
        # 对于其余画图的frame，也进行相同的操作
        self.editor.frameInTraining.setLayout(QVBoxLayout()) 
        self.editor.frameInPrediction.setLayout(QVBoxLayout())
        self.editor.frameInDetection.setLayout(QVBoxLayout())
        self.canvasInTraining = FigureCanvas(Figure())
        self.canvasInPrediction = FigureCanvas(Figure()) 
        self.canvasInDetection = FigureCanvas(Figure())
        self.editor.frameInTraining.layout().addWidget(self.canvasInTraining)
        self.editor.frameInPrediction.layout().addWidget(self.canvasInPrediction)
        self.editor.frameInDetection.layout().addWidget(self.canvasInDetection)
        self.canvasInTraining.installEventFilter(self)
        self.canvasInPrediction.installEventFilter(self)
        self.canvasInDetection.installEventFilter(self)

        # Menu bar to set parameters
        menu_bar = self.menuBar() # 创建菜单栏
        settings_menu = menu_bar.addMenu("Settings") # 添加一个菜单
        set_params_action = QAction("Set Parameters", self) # 添加一个动作
        set_params_action.triggered.connect(self.set_parameters) # 动作绑定事件
        settings_menu.addAction(set_params_action) # 将动作添加到菜单中

        self.cfg = cfg_GUI
        self.model = None
        self.refence_errors = np.array([])

        logger.info("GUI window initialized")
        set_random_seed(self.cfg.SEED)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        if(self.cfg.LOG.OUTPUT_TO_FILE):
            logger.add(self.cfg.LOG.DIR + f'/{self.cfg.LOG.PREFIX}_{cur_time}.log', encoding='utf-8')
    
    def set_parameters(self):
        # 设置参数的事件函数
        dialog = SetParametersDialog(self, self.cfg)
        if dialog.exec_():
            self.cfg = dialog.getValues()
            logger.info("Parameters set: {}".format(self.cfg))

    def eventFilter(self, source, event):
        # 检查事件类型和事件来源是否为 FigureCanvas
        if isinstance(source, FigureCanvas) \
                and event.type() == QEvent.MouseButtonDblClick \
                and event.button() == Qt.LeftButton:
            logger.info("Double click detected on canvas")
            self.show_in_new_window(source)
            return True
        # logger.info("Event not handled")
        return super().eventFilter(source, event)
    
    def show_in_new_window(self, canvas):
        """
        在全屏的新窗口中显示复制的图形
        """
        # 保存原始 figure 到临时文件
        figure = canvas.figure
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            figure.savefig(temp_file.name, dpi=300)

        # 打开一个最大化的新窗口并保持引用
        self.enlarged_window = EnlargedWindow(temp_file.name)
        self.enlarged_window.showMaximized()
    
    @pyqtSlot() #导入正常信号 for 特征筛选
    def on_btnImportNormalSignalInSelection_clicked(self):
        # open multiple files with the .csv extension
        file_paths, _ = QFileDialog.getOpenFileNames(self, "导入正常信号", "./", "Comma-Separated Values (*.csv)")
        # Check if any files were selected
        if not file_paths:
            logger.info("未选择任何文件")
            return
        
        self.cfg.FEATURE.NORMAL_PATHS = [] # set empty
        for fname in file_paths:
            # Check if the file has a .csv extension
            if not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
            logger.info(f"Normal signal imported (in feature selection): {fname}")
            self.cfg.FEATURE.NORMAL_PATHS.append(fname)

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

    @pyqtSlot() #导入未知信号 for 新数据预测
    def on_btnImportUnknownSignal_clicked(self):
        fname,_ = QFileDialog.getOpenFileName(self, "导入未知信号","./", "Comma-Separated Values(*.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        logger.info("Unknown signal imported: {}".format(fname))
        self.cfg.INFERENCE.UNKWON_PATH = fname
    
    @pyqtSlot() #导入正常信号 for 故障检测
    def on_btnImportInferentSignal_Dete_clicked(self):
        fname,_ = QFileDialog.getOpenFileName(self, "导入正常信号","./", "Comma-Separated Values(*.csv)")
        if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        if self.cfg.TRAIN.NORMAL_PATH and Path(self.cfg.TRAIN.NORMAL_PATH).exists():
            checkAndWarn(self, fname == self.cfg.TRAIN.NORMAL_PATH, 
                         false_fb="导入的正常信号与过往导入的正常信号不一致，若坚持使用不一致的数据，请关闭该警告窗口")
        logger.info("Normal signal imported: {}".format(fname))
        self.cfg.TRAIN.NORMAL_PATH = fname
    
    @pyqtSlot() #导入未知信号集 for 故障检测
    def on_btnImportFullTest_clicked(self):
        fname = QFileDialog.getExistingDirectory(self, "导入未知信号集","./")
        # if fname and not checkAndWarn(self,fname[-4:]=='.csv',false_fb="选中的文件并非.csv类型，请检查"): return
        logger.info("Unknown signal directory imported: {}".format(fname))
        self.cfg.INFERENCE.TEST_CONTENT = fname

    @pyqtSlot() #导入预测模型 for 故障诊断
    def on_btnImportModel_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入预测模型","./", "PyTorch model(*.pth)")
        # if fname: logger.info("Model imported: name {}, type {}".format(fname, ftype))
        if fname and not checkAndWarn(self,fname[-4:]=='.pth',false_fb="选中的文件并非.pth类型，请检查"): return
        if(fname): 
            logger.info("Model imported: {}".format(fname))
            self.model = tload(fname)
    
    @pyqtSlot() #导入预测模型 for 故障检测
    def on_btnImportModel_Dete_clicked(self):
        fname,ftype = QFileDialog.getOpenFileName(self, "导入预测模型","./", "PyTorch model(*.pth)")
        logger.info("Model imported: name {}, type {}".format(fname, ftype))
        if fname and not checkAndWarn(self,fname[-4:]=='.pth',false_fb="选中的文件并非.pth类型，请检查"): return
        if(fname): 
            logger.info("Model imported: {}".format(fname))
            self.model = tload(fname)
    
    @pyqtSlot() # 计算DTW并展示
    def on_btnCalculateDTW_clicked(self):
        # 检查是否导入了正常与故障信号
        state = self.cfg.FEATURE.NORMAL_PATHS and self.cfg.TRAIN.FAULT_PATH
        if not checkAndWarn(self,state,
                            "数据导入成功，开始计算",
                            "数据缺失，请导入正常与故障信号",
                            True): return

        # 计算DTW得分
        logger.info("Search propre features...")
        try:
            ranked_feat, feat_df = view_features_DTW_with_n_normal(
                    normal_paths= self.cfg.FEATURE.NORMAL_PATHS,
                    fault_path= self.cfg.TRAIN.FAULT_PATH,
                    feat_max_length= self.cfg.FEATURE.MAX_LENGTH,
                    need_denoise= self.cfg.DENOISE.NEED,
                    denoise_method= self.cfg.DENOISE.METHOD,
                    smooth_step= self.cfg.DENOISE.SMOOTH_STEP,
                    wavelet= self.cfg.DENOISE.WAVELET, 
                    level=self.cfg.DENOISE.LEVEL,
                    channel_score_mode= self.cfg.FEATURE.CHANNEL_SCORE_MODE,
            ) # ranked_feat is list[tuple[str, float]], feat_df is pd.DataFrame
        except Exception as e:
            logger.error(f"meet error {e} in view_features_DTW_with_n_normal")
            checkAndWarn(self,False,false_fb="计算DTW得分时遇到错误，请检查控制台信息")
            return 
            
        logger.info("features ranked:\n{}".format('\n'.join(f"{k}: {v}" for k, v in ranked_feat)))

        # 绘制热力图
        logger.info("Draw heat map...")
        total_corr = feat_df.corr()
        chn = len(feat_df.columns)//18
        col = [item.split('_')[1] for item in total_corr.columns[:18]] # 从列名中提取特征名
        corr_values = np.zeros((18, 18)) # 使用 NumPy 数组进行累加
        for i in range(chn):
            sub_corr = total_corr.iloc[i*18:(i+1)*18, i*18:(i+1)*18].values  # 每个通道的相关性，使用.values获取NumPy数组
            corr_values += sub_corr
        corr = pd.DataFrame(corr_values / chn, columns=col, index=col) # 计算平均相关性
        draw_heatmap(corr, self.canvasInFeatures)

        # 找出相关性高的特征对并显示
        logger.info("Find correlated features...")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.8:
                    corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j])) 
        df = pd.DataFrame(corr_pairs, columns=["feature1", "feature2", "correlation"])
        df.sort_values(by='correlation', ascending=False, inplace=True)
        show_table_in_widget(df, self.editor.CorrWidget)

        # 将排序后的列表转换为 Pandas DataFrame
        df = pd.DataFrame(ranked_feat, columns=["feature", "DTW score"])
        # 在 QTableWidget 中显示 DataFrame
        show_table_in_widget(df, self.editor.DTWWidget)

        # 重设下拉多选框的选项
        self.editor.comboBoxSelectFeaturesInTraining.clear()
        self.editor.comboBoxSelectFeaturesInTraining.addItems([i[0] for i in ranked_feat])
        self.editor.comboBoxSelectFeaturesInTraining.selectItems([0]) # 默认选中第一个
        self.editor.comboBoxSelectFeaturesInPrediction.clear()
        self.editor.comboBoxSelectFeaturesInPrediction.addItems([i[0] for i in ranked_feat])
        self.editor.comboBoxSelectFeaturesInPrediction.selectItems([0])
        self.editor.comboBoxSelectFeaturesInDetection.clear()
        self.editor.comboBoxSelectFeaturesInDetection.addItems([i[0] for i in ranked_feat])
        self.editor.comboBoxSelectFeaturesInDetection.selectItems([0])

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
        # 检查正常信号的长度是否符合cfg要求
        normal_data = pd.read_csv(self.cfg.TRAIN.NORMAL_PATH)
        n,_ = normal_data.shape
        if not n > self.cfg.DESIGN.PIECE + self.cfg.DESIGN.SUBLEN:
            logger.warning(f"正常信号长度{n}不符合要求，至少需要{self.cfg.DESIGN.PIECE + self.cfg.DESIGN.SUBLEN}个采样点")
            self.cfg.DESIGN.FPIECE = (n - self.cfg.DESIGN.SUBLEN) // self.cfg.DESIGN.FSUBLEN
            self.cfg.DESIGN.PIECE = self.cfg.DESIGN.FPIECE * self.cfg.DESIGN.FSUBLEN
            logger.warning(f"已重设fpiece为{self.cfg.DESIGN.FPIECE}，piece为{self.cfg.DESIGN.PIECE}")
        
        # 开始训练
        logger.info("Start training with config:{}".format(self.cfg))
        logger.info("feature(s) used: {}".format(', '.join(self.cfg.FEATURE.USED_F)))
        self.model = set_train_model(self.cfg)

        # 计算阈值
        logger.info('Start to calculate threshold and distribution...')
        normal_errors = raw_signal_to_errors(self.cfg, self.model, is_normal=True)
        self.refence_errors = normal_errors

        # 绘制并显示
        hist_tied_to_frame(self.cfg,normal_errors,self.canvasInTraining,is_train=True)
 
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
        logger.info("Start predicting...")
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
        # 读取模型的输入维数并检查
        input_dim = self.model.lstm.input_size #模型的输入维数/通道数
        data_tunnel = pd.read_csv(self.cfg.INFERENCE.UNKWON_PATH).shape[1] #数据表列数
        feature_n = len(self.cfg.FEATURE.USED_F)
        state = (feature_n * data_tunnel == input_dim)
        if not checkAndWarn(self,state,false_fb=f'模型输入维数{input_dim}与数据通道数{data_tunnel}、特征数{feature_n}不匹配'): return

        logger.info("Set Prediction Thread...")
        self.predict_thread = PredictThread(self.cfg,
                                       self.model,
                                       self.refence_errors)     #配置线程
        self.predict_thread.signal_finish.connect(self.PredictionDraw) #绑定信号函数
        logger.info("Start prediction thread...")
        self.predict_thread.start()  # 启动线程

    def PredictionDraw(self, errs):
        # errs = [self.refence_errors, errors]
        hist_tied_to_frame(self.cfg, errs, 
                           self.canvasInPrediction,
                           is_train=False)
    
    @pyqtSlot() # 对全寿命数据进行故障检测
    def on_btnDetect_clicked(self):
        logger.info("Start detecting...")
        logger.info("Start checkout config matching...")
        # 检测是否有未知数据
        if not checkAndWarn(self,self.cfg.INFERENCE.TEST_CONTENT,false_fb="请导入待测信号"): return
        # 检测是否有正常信号
        if not checkAndWarn(self,self.cfg.TRAIN.NORMAL_PATH,false_fb="请导入作为参考的正常信号"): return
        # 检测是否有模型
        if not checkAndWarn(self,self.model,false_fb="请导入预测模型"): return
        # 检测是否选择特征
        pointed_features = self.editor.comboBoxSelectFeaturesInDetection.currentData()
        if pointed_features: self.cfg.FEATURE.USED_F = pointed_features
        if not checkAndWarn(self,self.cfg.FEATURE.USED_F,false_fb="未选中任何特征"): return
        # 检查比例阈值是否正常
        if not checkAndWarn(self,0 <= self.cfg.INFERENCE.MAE_ratio_threshold <= 1,
                            false_fb="比例阈值应在0-1之间"): return
        # 读取模型的输入维数并检查
        cont = Path(self.cfg.INFERENCE.TEST_CONTENT)
        logger.info('Full roll test data directory: {}'.format(cont))
        files = sort_list(list(cont.glob('*.csv')))
        input_dim = self.model.lstm.input_size #模型的输入维数/通道数
        data_channel = pd.read_csv(files[0]).shape[1] #数据表列数
        feature_n = len(self.cfg.FEATURE.USED_F)
        state = (feature_n * data_channel == input_dim)
        if not checkAndWarn(self, state,
                            false_fb=f'模型输入维数{input_dim}与数据通道数{data_channel}、特征数{feature_n}不匹配'): return

        
        logger.info("Set Detecion Thread...")
        self.detect_thread = DetectThread(self.cfg, 
                                         self.model, 
                                         self.refence_errors)     #配置线程
        self.detect_thread.signal_turn.connect(self.DetectionTurnDraw) #绑定信号函数
        logger.info("Start detection thread...")
        self.detect_thread.start()  # 启动线程
        logger.info("Thread finished")
    
    @pyqtSlot() # 停止检测按钮
    def on_btnStopDetect_clicked(self):
        if hasattr(self, 'detect_thread'):
            self.detect_thread.stop()
            logger.info("Detection stopped by user")
        else:
            logger.info("No detection thread to stop")

    def DetectionTurnDraw(self, res):
        res_series = pd.Series({k: v for k, v in res.items()})
        update_ratio_to_frame(self.cfg, res_series, self.canvasInDetection)

#%% 开始运行

def main():
    app = QApplication(sys.argv)
    # app.setQuitOnLastWindowClosed(True) #添加这行才能在spyder正常退出
    w = GUIWindow()
    w.show()
    n = app.exec()
    sys.exit(n)

if __name__ == '__main__':
    main()