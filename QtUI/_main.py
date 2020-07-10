import main
import sys
from PyQt5 import QtCore, QtGui, QtWidgets



class MainWindow(QtWidgets.QMainWindow, main.Ui_MainWindow):

    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        # MainWindow = QtWidgets.QMainWindow()
        QtWidgets.QMainWindow.__init__(self)

        self.setupUi(self)
        self.show()
        sys.exit(app.exec_())

    def textchanged(self):
        print("the text was changed")

if __name__ == "__main__":
    mainw = MainWindow()



