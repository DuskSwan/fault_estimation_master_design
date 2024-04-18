import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print("Qt: v", QT_VERSION_STR, "\tPyQt: v", PYQT_VERSION_STR)

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        button = QPushButton("Press Me!")

        # Set the central widget of the Window.
        self.setCentralWidget(button)

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r''

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()