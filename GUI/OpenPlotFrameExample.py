import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QDialog, QGridLayout
from PyQt5.QtCore import Qt, QEvent
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('PyQt5 Double Click Plot Example')
        self.setGeometry(100, 100, 800, 600)

        # 创建一个QFrame作为中央部件
        self.frame = QFrame(self)
        self.frame.setLayout(QVBoxLayout())
        self.setCentralWidget(self.frame)

        # 在QFrame中创建Matplotlib图形
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.frame.layout().addWidget(self.canvas)

        # 在画布上绘制示例图形
        self.ax = self.figure.add_subplot(111)
        self.ax.hist([10, 20, 30, 40, 50, 60, 70, 80, 90], bins=10, color='blue', label='Histogram')
        self.ax.legend()

        # 为 FigureCanvas 设置鼠标双击事件过滤器
        self.canvas.installEventFilter(self)

    def eventFilter(self, source, event):
        # 如果事件来源是 FigureCanvas 且事件类型是鼠标双击
        if source == self.canvas and event.type() == QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:
                self.show_in_new_window()
            return True
        return super(MainWindow, self).eventFilter(source, event)

    def show_in_new_window(self):
        # 创建并打开一个新的对话框窗口，以显示较大的图
        dialog = QDialog(self)
        dialog.setWindowTitle("Larger Plot")
        dialog.setLayout(QGridLayout())

        # 为对话框创建一个较大的画布，设置较大的 figsize
        figure = Figure(figsize=(10, 8))  # 调整画布大小（单位：英寸）
        canvas = FigureCanvas(figure)
        dialog.layout().addWidget(canvas, 0, 0)

        # 在新画布上重新绘制当前图形
        ax = figure.add_subplot(111)
        ax.hist([10, 20, 30, 40, 50, 60, 70, 80, 90], bins=10, color='blue', label='Histogram')
        ax.legend()

        # 显示新的对话框
        dialog.resize(1000, 800)  # 调整新窗口的大小
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
