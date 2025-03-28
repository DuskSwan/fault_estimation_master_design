# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate, qApp
from PyQt5.QtGui import QFontMetrics, QPalette
from PyQt5.QtCore import QEvent, Qt
from PyQt5.Qt import QStandardItem

class CheckableComboBox(QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):
        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def clearItems(self):
        self.model().clear()

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res
    
    def selectItems(self,idxs):
        for i in idxs:
            if i < self.model().rowCount():
                self.model().item(i).setCheckState(Qt.Checked)
        

'''
Usage:
comunes = ['RMS','SRA', 'KV', 'SV', 'PPV',
         'CF', 'IF', 'MF', 'SF', 'KF',
         'FC', 'RMSF', 'RVF',
         'Mean', 'Var', 'Std', 'Max', 'Min',
         ]
combo = CheckableComboBox()
combo.addItems(comunes)
'''

# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QPushButton

if __name__ == '__main__':
    app = QApplication([])
    window = QMainWindow()
    central_widget = QWidget()
    layout = QVBoxLayout()
    
    combo = CheckableComboBox()
    comunes = ['RMS', 'SRA', 'KV', 'SV', 'PPV',
               'CF', 'IF', 'MF', 'SF', 'KF',
               'FC', 'RMSF', 'RVF',
               'Mean', 'Var', 'Std', 'Max', 'Min']
    combo.addItems(comunes)
    combo.selectItems([0, 1])  # 选中前两个项
    
    layout.addWidget(combo)
    
    btnShow = QPushButton("Show Selection")
    btnShow.clicked.connect(lambda: print(combo.currentData()))
    layout.addWidget(btnShow)

    btnClear = QPushButton("Clear item")
    btnClear.clicked.connect(lambda: combo.clearItems())
    layout.addWidget(btnClear)
    
    central_widget.setLayout(layout)
    window.setCentralWidget(central_widget)
    
    window.show()
    app.exec_()
