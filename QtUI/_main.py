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

        self.COLORGREEN='#54f542'

        self.setupUi(self)

        # raw image
        self.display_raw_draw = {}
        self.display_raw_draw["data"] = pg.ImageItem()
        self.display_raw_draw["plot"] = self.display_raw.addPlot()
        self.display_raw_draw["plot"].addItem(self.display_raw_draw["data"])
        self.display_raw_draw["plot"].getAxis('left').setLabel('Pixel', color=self.COLORGREEN)
        self.display_raw_draw["plot"].getAxis('bottom').setLabel('Pixel', color=self.COLORGREEN)
        self.display_raw_draw["plot"].setTitle('Raw',color=self.COLORGREEN)

        # processed image
        self.display_proc_draw = {}
        self.display_proc_draw["data"] = pg.ImageItem()
        self.display_proc_draw["plot"] = self.display_proc.addPlot()
        self.display_proc_draw["plot"].addItem(self.display_proc_draw["data"])
        self.display_proc_draw["plot"].getAxis('left').setLabel('Spatial Frequency', color=self.COLORGREEN)
        self.display_proc_draw["plot"].getAxis('bottom').setLabel('Spatial Frequency', color=self.COLORGREEN)
        self.display_proc_draw["plot"].setTitle('Processed',color=self.COLORGREEN)

        # reconstructed
        self.display_recons_draw = {}
        self.display_recons_draw["data"] = pg.ImageItem()
        self.display_recons_draw["plot"] = self.display_recons.addPlot()
        self.display_recons_draw["plot"].addItem(self.display_recons_draw["data"])
        self.display_recons_draw["plot"].getAxis('left').setLabel('Spatial Frequency', color=self.COLORGREEN)
        self.display_recons_draw["plot"].getAxis('bottom').setLabel('Spatial Frequency', color=self.COLORGREEN)
        self.display_recons_draw["plot"].setTitle('Reconstructed',color=self.COLORGREEN)

        # intensity / real
        self.display_intens_real_draw = {}
        self.display_intens_real_draw["data"] = pg.ImageItem()
        self.display_intens_real_draw["plot"] = self.display_intens_real.addPlot()
        self.display_intens_real_draw["plot"].addItem(self.display_intens_real_draw["data"])
        self.display_intens_real_draw["plot"].getAxis('left').setLabel('Position', color=self.COLORGREEN)
        self.display_intens_real_draw["plot"].getAxis('bottom').setLabel('Position', color=self.COLORGREEN)
        self.display_intens_real_draw["plot"].setTitle('Intensity',color=self.COLORGREEN)

        # phase / imag
        self.display_phase_imag_draw = {}
        self.display_phase_imag_draw["data"] = pg.ImageItem()
        self.display_phase_imag_draw["plot"] = self.display_phase_imag.addPlot()
        self.display_phase_imag_draw["plot"].addItem(self.display_phase_imag_draw["data"])
        self.display_phase_imag_draw["plot"].getAxis('left').setLabel('Position', color=self.COLORGREEN)
        self.display_phase_imag_draw["plot"].getAxis('bottom').setLabel('Position', color=self.COLORGREEN)
        self.display_phase_imag_draw["plot"].setTitle('Phase',color=self.COLORGREEN)

        # state of UI
        self.running=False


        self.show()
        sys.exit(app.exec_())

    def textchanged(self):
        for _ in range(20):
            QtCore.QCoreApplication.processEvents()
            print("the text was changed")

    def Start_Stop_Clicked(self):
        if not self.running:
            self.running=True
            self.pushButton.setText("Stop")
            self.run_retrieval()

        if self.running:
            self.running=False
            self.pushButton.setText("Start")

    def run_retrieval(self):

        while self.running:
            QtCore.QCoreApplication.processEvents()
            self.display_proc_draw["data"].setImage(np.random.rand(10, 10) + 10.)
            self.display_proc_draw["plot"].getAxis('left').setLabel('some bullshit', color='#54f542')
            self.display_proc_draw["plot"].getAxis('bottom').setLabel('some other bullsht', color='#54f542')



if __name__ == "__main__":
    mainw = MainWindow()



