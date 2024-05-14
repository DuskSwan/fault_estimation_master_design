import sys
import tempfile
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QGridLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QEvent

class CustomFigureCanvas(FigureCanvas):
    def __init__(self, figure):
        super().__init__(figure)

class EnlargedWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        # 设置窗口标题并创建中央部件
        self.setWindowTitle("Enlarged Plot")
        central_widget = QFrame(self)
        central_widget.setLayout(QGridLayout())
        self.setCentralWidget(central_widget)

        # 创建一个新的 Figure 和 FigureCanvas
        new_figure = Figure(figsize=(24, 18))
        new_canvas = FigureCanvas(new_figure)
        central_widget.layout().addWidget(new_canvas, 0, 0)

        # 从临时文件中读取并绘制在新的 figure 上
        new_ax = new_figure.add_subplot(111)
        new_ax.axis('off')  # 关闭坐标轴
        new_ax.imshow(plt.imread(image_path))
        new_ax.set_xticks([])  # 关闭刻度
        new_ax.set_yticks([])
        new_ax.spines['top'].set_visible(False)  # 关闭边框
        new_ax.spines['bottom'].set_visible(False)
        new_ax.spines['left'].set_visible(False)
        new_ax.spines['right'].set_visible(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # 设置窗口标题和尺寸
        self.setWindowTitle('Enlarged Plot Example')
        self.setGeometry(100, 100, 1000, 800)

        # 创建一个 QFrame 并设置为中央部件
        self.frame = QFrame(self)
        self.frame.setLayout(QVBoxLayout())
        self.setCentralWidget(self.frame)

        # 创建 Figure 和 CustomFigureCanvas 对象
        self.figure1 = Figure()
        self.canvas1 = CustomFigureCanvas(self.figure1)
        self.frame.layout().addWidget(self.canvas1)

        # 安装事件过滤器到 CustomFigureCanvas 上
        self.canvas1.installEventFilter(self)

        # 在画布上绘制示例图形
        self.plot_example(self.figure1)

        # 用于保持新窗口的引用
        self.enlarged_window = None

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonDblClick and isinstance(source, CustomFigureCanvas):
            self.show_in_new_window(source.figure)
            return True
        return super(MainWindow, self).eventFilter(source, event)

    def plot_example(self, figure):
        """
        绘制示例图
        """
        ax = figure.add_subplot(111)
        ax.plot([0, 1, 2, 3], [10, 20, 25, 30], marker='o', label='Line Plot')
        ax.set_title('Original Plot')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.legend()
        figure.canvas.draw()

    def show_in_new_window(self, figure):
        """
        在最大化的新窗口中显示复制的图形
        """
        # 保存原始 figure 到临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            figure.savefig(temp_file.name, dpi=300)

        # 打开一个最大化的新窗口并保持引用
        self.enlarged_window = EnlargedWindow(temp_file.name)
        self.enlarged_window.showMaximized()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
