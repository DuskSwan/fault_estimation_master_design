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

def modify_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 在首行插入新的导入语句
    lines.insert(0, 'from .CheckableComboBoxPY import CheckableComboBox\n')
    lines.insert(1, "comunes = ['RMS','SRA', 'KV', 'SV', 'PPV', 'CF', 'IF', 'MF', 'SF', 'KF', 'FC', 'RMSF', 'RVF', 'Mean', 'Var', 'Std', 'Max', 'Min',]\n")

    # 替换语句
    inde = ' ' * 8 # 缩进数
    for i in range(len(lines)):
        # 训练 - 特征选择下拉框
        if 'self.comboBoxSelectFeaturesInTraining = QtWidgets.QComboBox(self.widgetInTriaining)' in lines[i]:
            lines[i] = inde + 'self.comboBoxSelectFeaturesInTraining = CheckableComboBox(self.widgetInTriaining)\n'
            lines.insert(i+1, inde + 'self.comboBoxSelectFeaturesInTraining.addItems(comunes)\n')
        # 预测 - 特征选择下拉框
        if 'self.comboBoxSelectFeaturesInPrediction = QtWidgets.QComboBox(self.widgetInPrediction)' in lines[i]:
            lines[i] = inde + 'self.comboBoxSelectFeaturesInPrediction = CheckableComboBox(self.widgetInPrediction)\n'
            lines.insert(i+1, inde + 'self.comboBoxSelectFeaturesInPrediction.addItems(comunes)\n')
        # 检测 - 特征选择下拉框
        if 'self.comboBoxSelectFeaturesInDetection = QtWidgets.QComboBox(self.widgetInDetection)' in lines[i]:
            lines[i] = inde + 'self.comboBoxSelectFeaturesInDetection = CheckableComboBox(self.widgetInDetection)\n'
            lines.insert(i+1, inde + 'self.comboBoxSelectFeaturesInDetection.addItems(comunes)\n')
        # 指定默认展示页
        if 'self.tabWidget.setCurrentIndex' in lines[i]:
            lines[i] = inde + 'self.tabWidget.setCurrentIndex(1)\n'
        # 指定窗口尺寸
        # <window name>.resize
        if 'MainWindow.resize' in lines[i]:
            lines[i] = inde + 'MainWindow.resize(1440, 960)\n'

    # 将修改后的内容写回新的文件名
    new_filename = filename.replace('.py', '_m.py')
    with open(new_filename, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# 使用函数
modify_file('GUI/Ui_MainWindow.py')