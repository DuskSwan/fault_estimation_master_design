from .CheckableComboBoxPY import CheckableComboBox
comunes = ['RMS','SRA', 'KV', 'SV', 'PPV', 'CF', 'IF', 'MF', 'SF', 'KF', 'FC', 'RMSF', 'RVF', 'Mean', 'Var', 'Std', 'Max', 'Min',]
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\GithubRepos\fault_estimation_master_design\GUI\FaultDegreeGUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FaultDiagnosis(object):
    def setupUi(self, FaultDiagnosis):
        FaultDiagnosis.setObjectName("FaultDiagnosis")
        FaultDiagnosis.resize(1280, 960)
        self.gridLayout = QtWidgets.QGridLayout(FaultDiagnosis)
        self.gridLayout.setContentsMargins(9, 9, 9, 9)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(FaultDiagnosis)
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setObjectName("tabWidget")
        self.PrepareTab = QtWidgets.QWidget()
        self.PrepareTab.setObjectName("PrepareTab")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.PrepareTab)
        self.horizontalLayout.setContentsMargins(9, 9, 9, 9)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widgetInFeatures = QtWidgets.QWidget(self.PrepareTab)
        self.widgetInFeatures.setObjectName("widgetInFeatures")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widgetInFeatures)
        self.horizontalLayout_3.setContentsMargins(9, 9, 9, 9)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btnImportNormalSignalInSelection = QtWidgets.QPushButton(self.widgetInFeatures)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnImportNormalSignalInSelection.sizePolicy().hasHeightForWidth())
        self.btnImportNormalSignalInSelection.setSizePolicy(sizePolicy)
        self.btnImportNormalSignalInSelection.setObjectName("btnImportNormalSignalInSelection")
        self.horizontalLayout_2.addWidget(self.btnImportNormalSignalInSelection)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.btnImportFaultSignalInSelection = QtWidgets.QPushButton(self.widgetInFeatures)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnImportFaultSignalInSelection.sizePolicy().hasHeightForWidth())
        self.btnImportFaultSignalInSelection.setSizePolicy(sizePolicy)
        self.btnImportFaultSignalInSelection.setObjectName("btnImportFaultSignalInSelection")
        self.horizontalLayout_2.addWidget(self.btnImportFaultSignalInSelection)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.btnCalculateDTW = QtWidgets.QPushButton(self.widgetInFeatures)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnCalculateDTW.sizePolicy().hasHeightForWidth())
        self.btnCalculateDTW.setSizePolicy(sizePolicy)
        self.btnCalculateDTW.setObjectName("btnCalculateDTW")
        self.horizontalLayout_2.addWidget(self.btnCalculateDTW)
        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 3)
        self.horizontalLayout_2.setStretch(3, 1)
        self.horizontalLayout_2.setStretch(4, 3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.tableWidget = QtWidgets.QTableWidget(self.widgetInFeatures)
        self.tableWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout.addWidget(self.tableWidget)
        self.verticalLayout.setStretch(0, 3)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 30)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.horizontalLayout.addWidget(self.widgetInFeatures)
        self.tabWidget.addTab(self.PrepareTab, "")
        self.TrainTab = QtWidgets.QWidget()
        self.TrainTab.setObjectName("TrainTab")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.TrainTab)
        self.horizontalLayout_4.setContentsMargins(9, 9, 9, 9)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.widgetInTriaining = QtWidgets.QWidget(self.TrainTab)
        self.widgetInTriaining.setObjectName("widgetInTriaining")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widgetInTriaining)
        self.gridLayout_2.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem3, 1, 1, 1, 1)
        self.btnSaveModel = QtWidgets.QPushButton(self.widgetInTriaining)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnSaveModel.sizePolicy().hasHeightForWidth())
        self.btnSaveModel.setSizePolicy(sizePolicy)
        self.btnSaveModel.setObjectName("btnSaveModel")
        self.gridLayout_2.addWidget(self.btnSaveModel, 8, 0, 1, 2)
        self.btnTraining = QtWidgets.QPushButton(self.widgetInTriaining)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnTraining.sizePolicy().hasHeightForWidth())
        self.btnTraining.setSizePolicy(sizePolicy)
        self.btnTraining.setObjectName("btnTraining")
        self.gridLayout_2.addWidget(self.btnTraining, 6, 0, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.widgetInTriaining)
        self.label_4.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widgetInTriaining)
        self.label_3.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.comboBoxSelectFeaturesInTraining = CheckableComboBox(self.widgetInTriaining)
        self.comboBoxSelectFeaturesInTraining.addItems(comunes)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxSelectFeaturesInTraining.sizePolicy().hasHeightForWidth())
        self.comboBoxSelectFeaturesInTraining.setSizePolicy(sizePolicy)
        self.comboBoxSelectFeaturesInTraining.setObjectName("comboBoxSelectFeaturesInTraining")
        self.gridLayout_2.addWidget(self.comboBoxSelectFeaturesInTraining, 2, 1, 1, 1)
        self.btnImportNormalSignalInTraining = QtWidgets.QPushButton(self.widgetInTriaining)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnImportNormalSignalInTraining.sizePolicy().hasHeightForWidth())
        self.btnImportNormalSignalInTraining.setSizePolicy(sizePolicy)
        self.btnImportNormalSignalInTraining.setObjectName("btnImportNormalSignalInTraining")
        self.gridLayout_2.addWidget(self.btnImportNormalSignalInTraining, 0, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem4, 7, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem5, 5, 0, 1, 1)
        self.horizontalLayout_4.addWidget(self.widgetInTriaining)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem6)
        self.frameInTraining = QtWidgets.QFrame(self.TrainTab)
        self.frameInTraining.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameInTraining.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameInTraining.setObjectName("frameInTraining")
        self.horizontalLayout_4.addWidget(self.frameInTraining)
        self.horizontalLayout_4.setStretch(0, 8)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 10)
        self.tabWidget.addTab(self.TrainTab, "")
        self.PredictTab = QtWidgets.QWidget()
        self.PredictTab.setObjectName("PredictTab")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.PredictTab)
        self.horizontalLayout_5.setContentsMargins(9, 9, 9, 9)
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.widgetInPrediction = QtWidgets.QWidget(self.PredictTab)
        self.widgetInPrediction.setObjectName("widgetInPrediction")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widgetInPrediction)
        self.gridLayout_3.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem7, 3, 1, 1, 1)
        self.btnPredict = QtWidgets.QPushButton(self.widgetInPrediction)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnPredict.sizePolicy().hasHeightForWidth())
        self.btnPredict.setSizePolicy(sizePolicy)
        self.btnPredict.setObjectName("btnPredict")
        self.gridLayout_3.addWidget(self.btnPredict, 6, 0, 1, 2)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem8, 1, 1, 1, 1)
        self.btnImportSignalInPrediction = QtWidgets.QPushButton(self.widgetInPrediction)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnImportSignalInPrediction.sizePolicy().hasHeightForWidth())
        self.btnImportSignalInPrediction.setSizePolicy(sizePolicy)
        self.btnImportSignalInPrediction.setObjectName("btnImportSignalInPrediction")
        self.gridLayout_3.addWidget(self.btnImportSignalInPrediction, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.widgetInPrediction)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 2, 0, 1, 1)
        self.btnImportModel = QtWidgets.QPushButton(self.widgetInPrediction)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnImportModel.sizePolicy().hasHeightForWidth())
        self.btnImportModel.setSizePolicy(sizePolicy)
        self.btnImportModel.setObjectName("btnImportModel")
        self.gridLayout_3.addWidget(self.btnImportModel, 2, 1, 1, 1)
        self.comboBoxSelectFeaturesInPrediction = CheckableComboBox(self.widgetInPrediction)
        self.comboBoxSelectFeaturesInPrediction.addItems(comunes)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxSelectFeaturesInPrediction.sizePolicy().hasHeightForWidth())
        self.comboBoxSelectFeaturesInPrediction.setSizePolicy(sizePolicy)
        self.comboBoxSelectFeaturesInPrediction.setObjectName("comboBoxSelectFeaturesInPrediction")
        self.gridLayout_3.addWidget(self.comboBoxSelectFeaturesInPrediction, 4, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.widgetInPrediction)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 4, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widgetInPrediction)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem9, 5, 1, 1, 1)
        self.horizontalLayout_5.addWidget(self.widgetInPrediction)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem10)
        self.frameInPrediction = QtWidgets.QFrame(self.PredictTab)
        self.frameInPrediction.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameInPrediction.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameInPrediction.setObjectName("frameInPrediction")
        self.horizontalLayout_5.addWidget(self.frameInPrediction)
        self.horizontalLayout_5.setStretch(0, 8)
        self.horizontalLayout_5.setStretch(1, 1)
        self.horizontalLayout_5.setStretch(2, 10)
        self.tabWidget.addTab(self.PredictTab, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)

        self.retranslateUi(FaultDiagnosis)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(FaultDiagnosis)

    def retranslateUi(self, FaultDiagnosis):
        _translate = QtCore.QCoreApplication.translate
        FaultDiagnosis.setWindowTitle(_translate("FaultDiagnosis", "FaultDetect"))
        self.btnImportNormalSignalInSelection.setText(_translate("FaultDiagnosis", "导入正常信号"))
        self.btnImportFaultSignalInSelection.setText(_translate("FaultDiagnosis", "导入故障信号"))
        self.btnCalculateDTW.setText(_translate("FaultDiagnosis", "计算DTW得分"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.PrepareTab), _translate("FaultDiagnosis", "特征筛选"))
        self.btnSaveModel.setText(_translate("FaultDiagnosis", "保存模型"))
        self.btnTraining.setText(_translate("FaultDiagnosis", "训练预测模型"))
        self.label_4.setText(_translate("FaultDiagnosis", "选择特征"))
        self.label_3.setText(_translate("FaultDiagnosis", "正常信号"))
        self.btnImportNormalSignalInTraining.setText(_translate("FaultDiagnosis", "导入"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.TrainTab), _translate("FaultDiagnosis", "模型训练"))
        self.btnPredict.setText(_translate("FaultDiagnosis", "预测"))
        self.btnImportSignalInPrediction.setText(_translate("FaultDiagnosis", "导入"))
        self.label_12.setText(_translate("FaultDiagnosis", "预测模型"))
        self.btnImportModel.setText(_translate("FaultDiagnosis", "导入"))
        self.label_13.setText(_translate("FaultDiagnosis", "选择特征"))
        self.label_5.setText(_translate("FaultDiagnosis", "待测信号"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.PredictTab), _translate("FaultDiagnosis", "新数据预测"))