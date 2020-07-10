import main
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg



class MainWindow(QtWidgets.QMainWindow, main.Ui_MainWindow):

    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        # MainWindow = QtWidgets.QMainWindow()
        QtWidgets.QMainWindow.__init__(self)

        self.setupUi(self)

        # setup 2-d plot
        self.measured_draw = {}
        self.measured_draw["data"] = pg.ImageItem()
        self.measured_draw["plot"] = self.graphicsView.addPlot()
        self.measured_draw["plot"].addItem(self.measured_draw["data"])
        self.measured_draw["plot"].getAxis('left').setLabel('Frequency[Hz]', color='#54f542')
        self.measured_draw["plot"].getAxis('bottom').setLabel('Delay[fs]', color='#54f542')


        self.show()
        sys.exit(app.exec_())

    def textchanged(self):
        while True:
            QtCore.QCoreApplication.processEvents()
            self.measured_draw["data"].setImage(np.random.rand(10, 10) + 10.)
            print("the text was changed")

if __name__ == "__main__":
    mainw = MainWindow()



