# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:10:48 2023

@author: 24112
"""

#%% 转换实际要用的窗口成为代码

# import os

# current_cont = os.path.split(os.path.realpath(__file__))[0] #脚本所在目录
# uiname = os.path.join(current_cont,'FaultDegreeGUI2.ui') #合并文件名

# # print(current_cont)

# save_cont = os.path.join(current_cont,'pack') #输出目录
# pyname = os.path.join(save_cont,'FaultDegreeGUI.py')
# command = 'pyuic5 -o {py} {ui}'.format(py=pyname,ui=uiname)

# os.chdir(current_cont)
# os.system(command)

### 现在使用vscode的pyqt插件，直接生成py文件

#%% 修改多选下拉框
'''
插入 from .CheckableComboBoxPY import CheckableComboBox
comunes = ['RMS','SRA', 'KV', 'SV', 'PPV',
         'CF', 'IF', 'MF', 'SF', 'KF',
         'FC', 'RMSF', 'RVF',
         'Mean', 'Var', 'Std', 'Max', 'Min',
         ]

用段

self.comboBoxSelectFeaturesInTraining = CheckableComboBox(self.widgetInTriaining)
self.comboBoxSelectFeaturesInTraining.addItems(comunes)
替换
self.comboBoxSelectFeaturesInTraining = QtWidgets.QComboBox(self.widgetInTriaining)

同理用
self.comboBoxSelectFeaturesInPrediction = CheckableComboBox(self.widget_5)
self.comboBoxSelectFeaturesInPrediction.addItems(comunes)
替换
self.comboBoxSelectFeaturesInPrediction = QtWidgets.QComboBox(self.widget_5)
'''

#%% 一些测试

from torch import load as tload
import torch.nn as nn

lstm = tload(r'.\model\LSTM.pt')

if isinstance(lstm, nn.Module):
    first_module = next(lstm.modules())
    input_dim = first_module.in_features  # 输入维度
else:
    input_dim = lstm.in_features  # 输入维度