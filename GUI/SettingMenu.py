from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Parameter Settings Example")
        self.setGeometry(100, 100, 400, 300)
        
        # 创建菜单栏
        menu_bar = self.menuBar()
        
        # 创建设置菜单
        settings_menu = menu_bar.addMenu("Settings")
        
        # 创建设置Epoch的动作
        set_epoch_action = QAction("Set Epoch", self)
        set_epoch_action.triggered.connect(self.set_epoch)
        settings_menu.addAction(set_epoch_action)

    def set_epoch(self):
        # 这里只是展示一个信息框作为示例
        QMessageBox.information(self, "Set Epoch", "Here you can set the epoch value.")
        
# 应用程序入口
def main():
    app = QApplication([])
    ex = MainWindow()
    ex.show()
    app.exec_()

if __name__ == "__main__":
    main()
