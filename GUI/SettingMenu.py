from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox, QVBoxLayout, QPushButton, QWidget, QDialog, QLineEdit, QLabel, QFormLayout
from yacs.config import CfgNode as CN

class SetParametersDialog(QDialog):
    def __init__(self, parent, cfg):
        super().__init__(parent)
        self.cfg = cfg.clone()
        self.param = {
            "m": "DESIGN.M",
            "p": "DESIGN.P",
            "epoch": "SOLVER.MAX_EPOCHS",
            "batch_size": "TRAIN.BATCH_SIZE",
            "learning_rate": "SOLVER.BASE_LR",
            "ratio_threshold": "INFERENCE.MAE_ratio_threshold",
        }
        self.editLines = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Set Parameters")
        layout = QFormLayout()

        for k,v in self.param.items():
            value = eval(f"self.cfg.{v}")
            self.editLines[k] = QLineEdit(str(value))
            layout.addRow(QLabel(f"{k}:"), self.editLines[k])

        # OK and Cancel buttons
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        layout.addRow(self.ok_button, self.cancel_button)

        self.setLayout(layout)

    def getValues(self):
        for k in self.param.keys():
            v = self.editLines[k].text()
            exec(f"self.cfg.{self.param[k]} = {v}")
        return self.cfg
    
class testWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = CN()
        self.cfg.DESIGN = CN()
        self.cfg.SOLVER = CN()
        self.cfg.DESIGN.M = 100
        self.cfg.DESIGN.P = 10
        self.cfg.SOLVER.MAX_EPOCHS = 100
        self.initUI()
    def initUI(self):
        self.setWindowTitle("Parameter Display Example")
        self.setGeometry(100, 100, 400, 300)

        # Create layout and widget
        layout = QVBoxLayout()
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

        # Create buttons for displaying parameters
        self.btn_m = QPushButton(f"Show m: {self.cfg.DESIGN.M}", self)
        self.btn_m.clicked.connect(lambda: QMessageBox.information(self, "Value of m", f"m = {self.cfg.DESIGN.M}"))
        self.btn_p = QPushButton(f"Show p: {self.cfg.DESIGN.P}", self)
        self.btn_p.clicked.connect(lambda: QMessageBox.information(self, "Value of p", f"p = {self.cfg.DESIGN.P}"))
        self.btn_epoch = QPushButton(f"Show epoch: {self.cfg.SOLVER.MAX_EPOCHS}", self)
        self.btn_epoch.clicked.connect(lambda: QMessageBox.information(self, "Value of epoch", f"epoch = {self.cfg.SOLVER.MAX_EPOCHS}"))

        layout.addWidget(self.btn_m)
        layout.addWidget(self.btn_p)
        layout.addWidget(self.btn_epoch)

        # Menu bar to set parameters
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("Settings")
        set_params_action = QAction("Set Parameters", self)
        set_params_action.triggered.connect(self.set_parameters)
        settings_menu.addAction(set_params_action)

    def set_parameters(self):
        dialog = SetParametersDialog(self, self.cfg)
        if dialog.exec_():
            self.cfg = dialog.getValues()
            self.btn_m.setText(f"Show m: {self.cfg.DESIGN.M}")
            self.btn_p.setText(f"Show p: {self.cfg.DESIGN.P}")
            self.btn_epoch.setText(f"Show epoch: {self.cfg.SOLVER.MAX_EPOCHS}")

def test():
    app = QApplication([])
    ex = testWindow()
    ex.show()
    app.exec_()

if __name__ == "__main__":
    test()
