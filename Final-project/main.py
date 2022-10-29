
from ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import cameraThread

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    t=cameraThread.CameraThread(ui)
    app.aboutToQuit.connect(t.stop)
    t.start()
    MainWindow.show()
    sys.exit(app.exec_())
