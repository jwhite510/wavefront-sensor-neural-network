import main
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import os
from GetMeasuredDiffractionPattern import GetMeasuredDiffractionPattern

class Processing():
    def __init__(self):
        # string
        self.orientation=None
        # float
        self.rotation=0.0
        # float
        self.scale=1.0


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

        # initialize processing parameters
        self.processing=Processing()
        # set the buttons to these values
        self.rotation_edit.setText(str(self.processing.rotation))
        self.scale_edit.setText(str(self.processing.scale))
        self.orientation_edit.addItems(['None','Left->Right','Up->Down','Left->Right & Up->Down'])
        self.orientation_edit.setCurrentIndex(2) # default to up->down

        im=self.retrieve_raw_img()
        experimental_params = {}
        experimental_params['pixel_size'] = 27e-6 # [meters] with 2x2 binning
        experimental_params['z_distance'] = 16e-3 # [meters] distance from camera
        experimental_params['wavelength'] = 633e-9 #[meters] wavelength
        self.getMeasuredDiffractionPattern = GetMeasuredDiffractionPattern(N_sim=128,
                N_meas=np.shape(im)[0], # for calculating the measured frequency axis (not really needed)
                experimental_params=experimental_params)

        # state of UI
        self.running=False
        self.plot_RE_IM=False


        self.show()
        sys.exit(app.exec_())

    def textchanged(self):
        print("the text was changed")

    def ProcessingUpdated(self):

        try:
            new_rotation = float(self.rotation_edit.text())
            new_scale = float(self.scale_edit.text())
            new_orientation = self.orientation_edit.currentText()
            if new_scale <= 0:
                raise ValueError("scale must be greater than 0")

            self.update_processing_values(new_rotation,new_scale,new_orientation)

        except Exception as e:
            print(e)
            pass

    def update_processing_values(self,new_rotation,new_scale,new_orientation):
        self.processing.orientation=new_orientation
        self.processing.rotation=new_rotation
        self.processing.scale=new_scale

    def Start_Stop_Clicked(self):
        if not self.running:
            self.running=True
            self.pushButton.setText("Stop")
            self.run_retrieval()

        if self.running:
            self.running=False
            self.pushButton.setText("Start")

    def TogglePlotRE_IM(self):

        if self.plot_RE_IM == True:
            self.plot_RE_IM=False
            self.display_phase_imag_draw["plot"].setTitle('Phase',color=self.COLORGREEN)
            self.display_intens_real_draw["plot"].setTitle('Intensity',color=self.COLORGREEN)
            self.view_toggle.setText("Real/Imag")

        elif self.plot_RE_IM == False:
            self.plot_RE_IM=True
            self.display_phase_imag_draw["plot"].setTitle('Imaginary',color=self.COLORGREEN)
            self.display_intens_real_draw["plot"].setTitle('Real',color=self.COLORGREEN)
            self.view_toggle.setText("Phase/\nIntensity")

    def run_retrieval(self):

        while self.running:
            QtCore.QCoreApplication.processEvents()

            # grab raw image
            im = self.retrieve_raw_img()
            # process image

            transform={}
            transform["rotation_angle"]=self.processing.rotation
            transform["scale"]=self.processing.scale
            if self.processing.orientation == "None":
                transform["flip"]=None

            elif self.processing.orientation == "Left->Right":
                transform["flip"]="lr"

            elif self.processing.orientation == "Up->Down":
                transform["flip"]="ud"

            elif self.processing.orientation == "Left->Right & Up->Down":
                transform["flip"]="lrud"
            im_p = self.getMeasuredDiffractionPattern.format_measured_diffraction_pattern(im, transform)

            # input through neural network

            im_p=np.squeeze(im_p)

            # grab image with orientation, rotation, scale settings
            print("self.processing.orientation =>", self.processing.orientation)
            print("self.processing.rotation =>", self.processing.rotation)
            print("self.processing.scale =>", self.processing.scale)

            self.display_proc_draw["data"].setImage(im_p)

    def retrieve_raw_img(self):
        x=np.linspace(-1,1,500).reshape(1,-1)
        y=np.linspace(-1,1,500).reshape(-1,1)

        z = np.exp(-x**2 / 0.5) * np.exp(-y**2 / 0.5)
        return z
        # return np.random.rand(500,600)



if __name__ == "__main__":
    mainw = MainWindow()



