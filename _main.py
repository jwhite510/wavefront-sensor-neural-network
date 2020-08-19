import main
import time
from numpy import unravel_index
import diffraction_net
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import os
from GetMeasuredDiffractionPattern import GetMeasuredDiffractionPattern
import pickle
from live_capture import TIS

class Processing():
    def __init__(self):
        # string
        self.orientation=None
        # float
        self.rotation=0.0
        # float
        self.scale=1.0


class MainWindow(QtWidgets.QMainWindow, main.Ui_MainWindow):

    def __init__(self,params):
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
        experimental_params['pixel_size'] = params['pixel_size'] # [meters] with 2x2 binning
        experimental_params['z_distance'] = params['z_distance'] # [meters] distance from camera
        experimental_params['wavelength'] = params['wavelength'] #[meters] wavelength
        self.getMeasuredDiffractionPattern = GetMeasuredDiffractionPattern(N_sim=128,
                N_meas=np.shape(im)[0], # for calculating the measured frequency axis (not really needed)
                experimental_params=experimental_params)

        # state of UI
        self.running=False
        self.plot_RE_IM=False

        # # initialize neural network
        self.network=diffraction_net.DiffractionNet(params['network']) # load a pre trained network

        # initialize camera
        self.Tis = TIS.TIS("48710182", 640, 480, 30, False)
        self.Tis.Start_pipeline()  # Start the pipeline so the camera streams


        self.show()
        sys.exit(app.exec_())

    def __del__(self):
        # cleanup camera
        Tis.Stop_pipeline()


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
            time1=time.time()
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
            print("input through net:")
            time_a=time.time()
            out_recons = self.network.sess.run( self.network.nn_nodes["recons_diffraction_pattern"], feed_dict={self.network.x:im_p})
            out_real = self.network.sess.run( self.network.nn_nodes["real_out"], feed_dict={self.network.x:im_p})
            out_imag = self.network.sess.run( self.network.nn_nodes["imag_out"], feed_dict={self.network.x:im_p})
            time_b=time.time()
            print(time_b-time_a)

            out_real=np.squeeze(out_real)
            out_imag=np.squeeze(out_imag)
            out_recons=np.squeeze(out_recons)

            # calculate the intensity
            complex_obj = out_real + 1j * out_imag
            I = np.abs(complex_obj)**2
            m_index = unravel_index(I.argmax(), I.shape)
            phase_Imax = np.angle(complex_obj[m_index[0], m_index[1]])
            complex_obj *= np.exp(-1j * phase_Imax)
            obj_phase = np.angle(complex_obj)

            # not using the amplitude_mask, use the absolute value of the intensity
            nonzero_intensity = np.array(np.abs(complex_obj))
            nonzero_intensity[nonzero_intensity < 0.01*np.max(nonzero_intensity)] = 0
            nonzero_intensity[nonzero_intensity >= 0.01*np.max(nonzero_intensity)] = 1
            obj_phase *= nonzero_intensity

            im_p=np.squeeze(im_p)

            # grab image with orientation, rotation, scale settings
            print("self.processing.orientation =>", self.processing.orientation)
            print("self.processing.rotation =>", self.processing.rotation)
            print("self.processing.scale =>", self.processing.scale)

            self.display_raw_draw["data"].setImage(im)
            self.display_proc_draw["data"].setImage(im_p)
            self.display_intens_real_draw["data"].setImage(I)
            self.display_phase_imag_draw["data"].setImage(obj_phase)
            self.display_recons_draw["data"].setImage(out_recons)
            time2=time.time()
            print("total time:")
            print(time2-time1)


    def retrieve_raw_img(self):

        # with open("sample.p", "rb") as file:
            # obj = pickle.load(file)
        # obj=np.pad(obj,pad_width=300,mode="constant",constant_values=0)
        # x=np.linspace(-1,1,500).reshape(1,-1)
        # y=np.linspace(-1,1,500).reshape(-1,1)

        # z = np.exp(-x**2 / 0.5) * np.exp(-y**2 / 0.5)
        # return obj
        # return np.random.rand(500,600)
        im=self.Tis.Get_image()
        im=np.sum(im,axis=2)
        return im





if __name__ == "__main__":
    params={}
    params['pixel_size']=27e-6
    params['z_distance']=16e-3
    params['wavelength']=633e-9
    params['network']="noise_test_D_fixednorm_SQUARE6x6_VISIBLESETUP_NOCENTER_peak-50"
    params['network']="vis1_2_peak-50"
    mainw = MainWindow(params)



