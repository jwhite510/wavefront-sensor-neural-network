import main
import params as globalparams
import diffraction_functions
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
# from live_capture import TIS
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from typing import Optional
# from vimba import *


def nparray_to_axislabel(arr:np.array,ticknumber:int)->list:
    axislabel=np.linspace(0,len(arr),ticknumber)
    if axislabel[-1]>=len(arr):axislabel[-1]-=1;
    axislabel=[[(int(v),"%.2f"%(arr[int(v)]))for v in axislabel]]
    return axislabel

def randomgaussiansignal()->np.array:
    x=np.linspace(-1,1,128).reshape(-1,1);y=np.linspace(-1,1,128).reshape(1,-1);w=0.5;
    gau=np.exp(-(x**2/w))*np.exp(-(y**2)/w);
    gau+=np.random.rand(128*128).reshape(128,128);
    return gau;


def addimageitemplot(qtgraphics,title:str,color:str,lut,ticks:list=None):
    newplot = {}
    newplot["data"] = pg.ImageItem()
    newplot["data"].setLookupTable(lut)
    newplot["plot"] = qtgraphics.addPlot()
    newplot["plot"].addItem(newplot["data"])
    newplot["plot"].getAxis('left').setLabel('Position', color=color)
    if ticks:newplot["plot"].getAxis('left').setTicks(ticks)
    if ticks:newplot["plot"].getAxis('bottom').setTicks(ticks)
    newplot["plot"].getAxis('bottom').setLabel('Position', color=color)
    newplot["plot"].setTitle(title,color=color)
    return newplot


def addimageviewplot(title:str,color:str,ticks:list=None,
        axislabel:str=None,
        ):

    # processed image
    leftaxis=pg.AxisItem(orientation='left')
    if ticks:leftaxis.setTicks(ticks)
    bottomaxis=pg.AxisItem(orientation='bottom')
    if ticks:bottomaxis.setTicks(ticks)
    plot=pg.PlotItem(axisItems={'left':leftaxis,'bottom':bottomaxis})

    if axislabel:plot.setLabel(axis='left',text=axislabel,color=color)
    if axislabel:plot.setLabel(axis='bottom',text=axislabel,color=color)
    plot.setTitle(title,color=color)
    widget = pg.ImageView(view=plot)
    colors = [
        (0, 0, 0),
        (45, 5, 61),
        (84, 42, 55),
        (150, 87, 60),
        (208, 171, 141),
        (255, 255, 255)
    ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    widget.setColorMap(cmap)
    # self.display_proc_draw["data"].setLookupTable(lut)
    # GraphicsLayoutWidget
    return widget



def print_preamble():
    print('//////////////////////////////////////////')
    print('/// Vimba API Synchronous Grab Example ///')
    print('//////////////////////////////////////////\n')


def print_usage():
    print('Usage:')
    print('    python synchronous_grab.py [camera_id]')
    print('    python synchronous_grab.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]


# def get_camera(camera_id: Optional[str]) -> Camera:
#     with Vimba.get_instance() as vimba:
#         if camera_id:
#             try:
#                 return vimba.get_camera_by_id(camera_id)
# 
#             except VimbaCameraError:
#                 abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
# 
#         else:
#             cams = vimba.get_all_cameras()
#             if not cams:
#                 abort('No Cameras accessible. Abort.')
# 
#             return cams[0]


# def setup_camera(cam: Camera):
#     with cam:
#         # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
#         try:
#             cam.GVSPAdjustPacketSize.run()
# 
#             while not cam.GVSPAdjustPacketSize.is_done():
#                 pass
# 
#         except (AttributeError, VimbaFeatureError):
#             pass

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

        # # initialize neural network
        # self.network=diffraction_net.DiffractionNet(params['network'],"original") # load a pre trained network
        # N=self.network.get_data.N;
        N=128
        # get position and frequenxy axis
        simulation_axes, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(
                N, int(globalparams.params.wf_ratio*N)
                )
        self.x=simulation_axes['object']['x'] # meters
        self.x*=1e6 # 1d numpy array [micrometers]

        self.f=simulation_axes['diffraction_plane']['f'] # 1/meters
        self.f*=1e-6 # 1d numpy array

        self.COLORGREEN='#54f542'

        self.setupUi(self)

        colormap = cm.get_cmap("nipy_spectral")
        colormap._init()
        lut = (colormap._lut*255).view(np.ndarray)


        # reconstructed image
        self.display_recons_draw=addimageviewplot('reconstruced',color=self.COLORGREEN,
                ticks=nparray_to_axislabel(self.f,3),
                axislabel='frequency [1/m] 10^6'
                );
        self.verticalLayout_3.addWidget(self.display_recons_draw)
        self.display_recons_draw.setImage(np.random.rand(128*128).reshape(128,128))

        # raw image
        self.display_raw_draw=addimageviewplot('raw image',color=self.COLORGREEN,axislabel='pixel');
        self.verticalLayout.addWidget(self.display_raw_draw)
        self.display_raw_draw.setImage(np.random.rand(128*128).reshape(128,128))

        # processed image
        self.display_proc_draw=addimageviewplot('processed image',color=self.COLORGREEN,
                ticks=nparray_to_axislabel(self.f,3),
                axislabel='frequency [1/m] 10^6'
                );
        self.verticalLayout.addWidget(self.display_proc_draw)
        self.display_proc_draw.setImage(np.random.rand(128*128).reshape(128,128))

        # intensity / real
        self.display_intens_real_draw=addimageviewplot('intensity',color=self.COLORGREEN,
                ticks=nparray_to_axislabel(self.x,3),
                axislabel='position [um]'
                )
        self.horizontalLayout_4.addWidget(self.display_intens_real_draw)
        self.display_intens_real_draw.setImage(np.random.rand(128*128).reshape(128,128))

        # phase / imag
        self.display_phase_imag_draw=addimageviewplot('phase[rad]',color=self.COLORGREEN,
                ticks=nparray_to_axislabel(self.x,3),
                axislabel='position [um]'
                )
        self.horizontalLayout_4.addWidget(self.display_phase_imag_draw)
        self.display_phase_imag_draw.setImage(np.random.rand(128*128).reshape(128,128))

        # initialize processing parameters
        self.processing=Processing()
        # set the buttons to these values
        self.rotation_edit.setText(str(self.processing.rotation))
        self.scale_edit.setText(str(self.processing.scale))
        self.orientation_edit.addItems(['None','Left->Right','Up->Down','Left->Right & Up->Down'])
        self.orientation_edit.setCurrentIndex(1) # default to Left->Right

        # initialize camera
        # self.Tis=params['Tis']
        # self.Tis.Start_pipeline()  # Start the pipeline so the camera streams

        # plt.ion()
        # while True:
            # if self.Tis.Snap_image(1) is True:  # Snap an image with one second timeout
                # image = self.Tis.Get_image()  # Get the image. It is a numpy array
                # plt.figure(1)
                # plt.imshow(np.squeeze(image))
                # plt.pause(0.1)
                # print("hello?")
        # self.Tis.Stop_pipeline()
        # exit()
        # im=None
        im=np.zeros((4000,4000),dtype=np.float32)
        # with Vimba.get_instance():
        #     with get_camera(None) as cam:
        #         setup_camera(cam)
        #         im=np.squeeze( cam.get_frame().as_numpy_ndarray() )

        # im=self.retrieve_raw_img()
        # self.Tis.Stop_pipeline()

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





        self.show()
        sys.exit(app.exec_())

    def __del__(self):
        pass
        # self.inst.close();
        # self.cam.close();
        # cleanup camera
        # self.Tis.Stop_pipeline()


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

        # with Vimba.get_instance():
            # with get_camera(None) as cam:
                # setup_camera(cam)
        while self.running:
            time1=time.time()
            QtCore.QCoreApplication.processEvents()

            # grab raw image
            # im = self.retrieve_raw_img()
            # process image
            # im=np.squeeze( cam.get_frame().as_numpy_ndarray() )
            im=np.random.rand(500*500).reshape(500,500)

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
            # out_recons = self.network.sess.run( self.network.nn_nodes["recons_diffraction_pattern"], feed_dict={self.network.x:im_p})
            out_recons=np.random.rand(128*128).reshape(128,128)
            # out_real = self.network.sess.run( self.network.nn_nodes["real_out"], feed_dict={self.network.x:im_p})
            out_real=np.random.rand(128*128).reshape(128,128)
            # out_imag = self.network.sess.run( self.network.nn_nodes["imag_out"], feed_dict={self.network.x:im_p})
            out_imag=np.random.rand(128*128).reshape(128,128)
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

            self.display_raw_draw.setImage(randomgaussiansignal(),
                autoRange=False,autoLevels=False,
                    )

            self.display_proc_draw.setImage(randomgaussiansignal(),
                autoRange=False,autoLevels=False,
                    )
            self.display_intens_real_draw.setImage(I)
            self.display_phase_imag_draw.setImage(obj_phase)

            self.display_recons_draw.setImage(randomgaussiansignal(),
                autoRange=False,autoLevels=False,
                    )

            time2=time.time()
            print("total time:")
            print(time2-time1)


    def retrieve_raw_img(self)->np.array:

        # with open("sample.p", "rb") as file:
            # obj = pickle.load(file)
        # obj=np.pad(obj,pad_width=300,mode="constant",constant_values=0)
        # x=np.linspace(-1,1,500).reshape(1,-1)
        # y=np.linspace(-1,1,500).reshape(-1,1)

        # z = np.exp(-x**2 / 0.5) * np.exp(-y**2 / 0.5)
        # return obj
        # return np.random.rand(500,600)

        # if self.Tis.Snap_image(1) is True:  # Snap an image with one second timeout
        #     im=self.Tis.Get_image()
        #     im=np.sum(im,axis=2)
        #     return im
        # else:
        #     return None
        # frame=self.cam.get_frame()
        # return np.squeeze(frame.to_np_ndarray())
        return None





if __name__ == "__main__":
    params={}
    params['pixel_size']=3.45e-6 # meters
    params['z_distance']=16.5e-3 # meter
    params['wavelength']=612e-9
    params['network']="varnoise_10ewfstest_2"

    # for camera
    # self.Tis = TIS.TIS("48710182", 640, 480, 30, False)
    # self.Tis = TIS.TIS("48710182", 2592, 2048, 60, False)
    # params['Tis']=TIS.TIS("48710182", 2592, 2048, 60, False)
    # https://github.com/TheImagingSource/tiscamera


    mainw = MainWindow(params)



