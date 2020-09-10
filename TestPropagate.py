import numpy as np
import diffraction_functions
import matplotlib.pyplot as plt
import pickle
from GetMeasuredDiffractionPattern import GetMeasuredDiffractionPattern
from PIL import Image, ImageDraw
import os


# def plot_beam(complex_b,title):
    # fig, ax = plt.subplots(2,1, figsize=(10,10))
    # fig.suptitle(title)
    # ax[0].pcolormesh(np.abs(complex_b))
    # ax[0].set_title("ABS")

def make_nice_figure(retrieved:dict):

    # get axes for retrieved object and diffraction pattern
    N=np.shape(np.squeeze(retrieved['0']['measured_pattern']))[0]
    simulation_axes, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
    # object
    x=simulation_axes['object']['x'] # meters
    x*=1e6
    f=simulation_axes['diffraction_plane']['f'] # 1/meters
    f*=1e-6

    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.02,wspace=0.02, left=0.1,right=0.8)
    # fig.text(0.5, 0.95, run_name, ha="center")
    # fig.subplots_adjust(hspace=0.0, left=0.2)
    gs = fig.add_gridspec(3,3)

    figletter = 'a'
    for _name, _i, _dist in zip(['-500','0','500_sim'],range(3),[-500,0,500]):
        complex_obj = np.squeeze(retrieved[_name]['real_output']+1j*retrieved[_name]['imag_output'])

        if _i == 0 or _i == 1:
            ax = fig.add_subplot(gs[_i,0])
            if _i==0:
                ax.set_title("Diffraction Pattern")
            ax.pcolormesh(f,f,np.squeeze(retrieved[_name]['measured_pattern']),cmap='jet')
            ax.text(0.04,0.9,figletter,transform=ax.transAxes,backgroundcolor='white',weight='bold')
            figletter = chr(ord(figletter)+1)
            # annotate measured
            ax.text(0.2, 0.9,'measured',backgroundcolor='white',transform=ax.transAxes)

            if _i == 1:
                ax.set_ylabel(r"frequency [1/m]$\cdot 10^{6}$")
            ax.set_xticks([])
            if _i == 0:
                ax.set_yticks([])

        ax = fig.add_subplot(gs[_i,1])
        if _i==0:
            ax.set_title("Intensity")
        ax.pcolormesh(x,x,np.abs(complex_obj)**2,cmap='jet')
        ax.text(0.04,0.9,figletter,transform=ax.transAxes,backgroundcolor='white',weight='bold')
        figletter = chr(ord(figletter)+1)
        ax.set_yticks([])
        if _i == 0 or _i == 1:
            ax.set_xticks([])
        if _i == 2:
            ax.set_xlabel(r"position [um]")
            ax.set_xticks([-5,-2.5,0,2.5,5])
        if _i == 0 or _i == 1:
            ax.text(0.2, 0.9,'retrieved',backgroundcolor='white',transform=ax.transAxes)
        if _i == 2:
            ax.text(0.2, 0.8,'propagated from\n -500 [um]',backgroundcolor='white',transform=ax.transAxes)
            # draw circle
            # circle to show where the wavefront originates
            circle=plt.Circle((0.6,-1.0),(2.7)/2,color='r',fill=False,linewidth=2.0)
            ax.add_artist(circle)
            ax.text(0.2, 0.4,"Spherical\nAperture\n2.7 um", fontsize=10, ha='center', transform=ax.transAxes,color="red",weight='bold')

        obj_phase = np.angle(complex_obj)
        # not using the amplitude_mask, use the absolute value of the intensity
        nonzero_intensity = np.array(np.abs(complex_obj))
        nonzero_intensity[nonzero_intensity < 0.05*np.max(nonzero_intensity)] = 0
        nonzero_intensity[nonzero_intensity >= 0.05*np.max(nonzero_intensity)] = 1
        obj_phase *= nonzero_intensity

        ax = fig.add_subplot(gs[_i,2])
        if _i ==0:
            ax.set_title("Phase")
        im=ax.pcolormesh(x,x,obj_phase,cmap='jet')
        ax.text(0.04,0.9,figletter,transform=ax.transAxes,backgroundcolor='white',weight='bold')
        figletter = chr(ord(figletter)+1)
        if _i == 0 or _i == 1:
            ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im,ax=ax)
        if _i == 2:
            ax.set_xlabel(r"position [um]")
            ax.set_xticks([-5,-2.5,0,2.5,5])
        ax.text(1.3,0.5,'z='+str(_dist)+' [um]',transform=ax.transAxes,size=20)
    fig.savefig('xuv_experimental_results_1.png')

    fig=plt.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.0,right=1.0,bottom=0.1)
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    im = Image.open('figure_images/xuv_experimental_setup.png')
    im = np.array(im)
    ax.imshow(im)
    ax.text(0.15,-0.1,r'z=500 [$\mu m$]',transform=ax.transAxes,size=20,ha='left')
    ax.text(0.45,-0.1,r'z=0 [$\mu m$]',transform=ax.transAxes,size=20,ha='left')
    ax.text(0.7,-0.1,r'z=-500 [$\mu m$]',transform=ax.transAxes,size=20,ha='left')
    ax.text(0.05,0.025,'z',transform=ax.transAxes,size=30,ha='left')

    # label WFS
    ax.text(0.55,1.05,'Wavefront Sensor',transform=ax.transAxes,size=15,ha='left')
    ax.text(0.60,1.0,'6 [$\mu m$]',transform=ax.transAxes,size=15,ha='left')
    ax.text(0.43,0.82,'6 [$\mu m$]',transform=ax.transAxes,size=15,ha='left')

    # label detector
    ax.text(0.57,0.2,'Detector',transform=ax.transAxes,size=15,ha='left')
    ax.text(0.88,0.2,'Detector',transform=ax.transAxes,size=15,ha='left')

    # label pinhole
    ax.text(385,337,'Pinhole',size=15,ha='left')
    ax.text(385,378,r'2.7 [$\mu m$]',size=15,ha='left')

    # label wavelength
    ax.text(105,365,r'18 [$nm$]',size=20,ha='left',color='black')
    ax.text(65,365,r'$\lambda$',size=20,ha='left',color='red')

    ax.axis('off')
    fig.savefig('xuv_experimental_results_2.png')

    fig=plt.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.0,right=1.0,bottom=0.1)
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    im = Image.open('figure_images/data_nn_diagram/datagen_nn.png')
    im = np.array(im)

    ax.text(52,150,'Gaussian',size=15,ha='left',color='black')
    ax.text(253,263,'Apply Phase',size=8,ha='left',color='black',backgroundcolor='white')

    ax.text(445,313,r'+',size=14,ha='left',color='black')
    ax.text(445,175,r'+',size=14,ha='left',color='black')

    ax.text(424,472,r'$e^{i \phi}$',size=14,ha='left',color='black',backgroundcolor='white')

    ax.text(560,263,'FFT',size=8,ha='left',color='black',backgroundcolor='white')
    ax.text(724,150,'Object',size=15,ha='left',color='black')

    ax.text(375,20,'Zernike\nCoefficients',size=15,ha='left',color='black')

    ax.text(337,542,'Beam Propagation Method',size=8,ha='left',color='black',backgroundcolor='white')

    ax.text(385,661,'FFT',size=8,ha='left',color='black',backgroundcolor='white')

    ax.text(733,669,'Neural\nNetwork',size=8,ha='left',color='black',backgroundcolor='white')

    ax.text(175,855,'Through\nWavefront\nSensor',size=15,ha='left',color='black')
    ax.text(522,835,'Diffraction\nPattern',size=15,ha='left',color='black')

    ax.text(883,835,'Retrieved\nObject',size=15,ha='left',color='black')

    # letters
    ax.text(39,206,'a',color='black',backgroundcolor='white',weight='bold')
    ax.text(387,59,'b',color='black',backgroundcolor='white',weight='bold')
    ax.text(686,206,'c',color='black',backgroundcolor='white',weight='bold')
    ax.text(155,607,'d',color='black',backgroundcolor='white',weight='bold')
    ax.text(505,607,'e',color='black',backgroundcolor='white',weight='bold')
    ax.text(858,607,'d',color='black',backgroundcolor='white',weight='bold')


    ax.imshow(im)
    ax.axis('off')
    fig.savefig('data_gen_simulation_3.png')

    fig=plt.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.0,right=1.0,bottom=0.1)
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    im = Image.open('figure_images/nn_diagram/nndiagram.png')
    im = np.array(im)
    print("np.shape(im)", np.shape(im))
    ax.imshow(im)
    ax.axis('off')

    # encoder
    ax.text(583, 485,'1',backgroundcolor='white',weight='bold')
    ax.text(707, 485,'2',backgroundcolor='white',weight='bold')
    ax.text(826, 485,'3',backgroundcolor='white',weight='bold')
    ax.text(948, 485,'4',backgroundcolor='white',weight='bold')


    # decoder
    ax.text(1070, 385,'5',backgroundcolor='white',weight='bold')
    ax.text(1194, 285,'6',backgroundcolor='white',weight='bold')
    ax.text(1313, 185,'7',backgroundcolor='white',weight='bold')

    # other side
    ax.text(1070, 605,'5',backgroundcolor='white',weight='bold')
    ax.text(1194, 705,'6',backgroundcolor='white',weight='bold')
    ax.text(1313, 805,'7',backgroundcolor='white',weight='bold')

    # real output
    ax.text(1433, 685,'Real Object',backgroundcolor='white',weight='bold')
    ax.text(1433, 85,'Imaginary Object',backgroundcolor='white',weight='bold')
    ax.text(70, 365,'Diffraction Pattern',backgroundcolor='white',weight='bold')

    # layers
    ax.text(400, -100,'Layers:\n \
            1: something\n \
            2: something\n \
            4: something\n \
            5: something\n \
            6: something\n \
            7: something\n \
            '
            ,backgroundcolor='white',size=15,ha='left',va='top')

    fig.savefig('nn_diagram_4.png')




    plt.show()
    print(retrieved.keys())


if __name__=="__main__":
    folder_dir="nn_pictures/teslatest5_doubleksize_doublefilters_reconscostfunction_pictures/46/measured/"
    # run_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    print("filename =>", filename)

    with open(filename,"rb") as file:
        obj=pickle.load(file)
    complex_beam=np.squeeze(obj["real_output"]+1j*obj["imag_output"])


    experimental_params = {}
    experimental_params['pixel_size'] = 27e-6 # [meters] with 2x2 binning
    experimental_params['z_distance'] = 33e-3 # [meters] distance from camera
    experimental_params['wavelength'] = 13.5e-9 #[meters] wavelength
    getMeasuredDiffractionPattern = GetMeasuredDiffractionPattern(N_sim=np.shape(complex_beam)[0],
            N_meas=2,#///
            experimental_params=experimental_params)

    f_object=getMeasuredDiffractionPattern.simulation_axes['diffraction_plane']['f']

    # distance to focus
    z = 1000e-6
    gamma=np.sqrt(
            1-
            (experimental_params['wavelength']*f_object.reshape(-1,1))**2-
            (experimental_params['wavelength']*f_object.reshape(1,-1))**2+
            0j # complex so it can be square rooted and imaginry
            )
    k_sq = 2 * np.pi * z / experimental_params['wavelength']
    # transfer function
    H = np.exp(1j * np.real(gamma) * k_sq) * np.exp(-1 * np.imag(gamma) * k_sq)

    complex_beam_f=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(complex_beam)))
    complex_beam_f*=H
    complex_beam_prop=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(complex_beam_f)))

    fig = diffraction_functions.plot_amplitude_phase_meas_retreival(
            {"measured_pattern":np.zeros_like(np.abs(complex_beam_prop)),
            "tf_reconstructed_diff":np.zeros_like(np.abs(complex_beam_prop)),
            "real_output":np.real(complex_beam_prop),
            "imag_output":np.imag(complex_beam_prop)},
            "z:-500 um -> + "+str(round(z*1e6,1))+" um [Simulated]",plot_spherical_aperture=True)
    # plt.savefig(str(_i)+"_SIMULATEDpropfrom-500z__"+str(round(z*1e6,1))+"_um_prop.png")
    plt.close(fig)
    print("saving:"+str(z*1e6)+"um")
    # plt.show()

    # create dictionary for publication figure
    retrieved = {}
    retrieved['500_sim'] = {
            "measured_pattern":np.zeros_like(np.abs(complex_beam_prop)),
            "tf_reconstructed_diff":np.zeros_like(np.abs(complex_beam_prop)),
            "real_output":np.real(complex_beam_prop),
            "imag_output":np.imag(complex_beam_prop)
            }


    # compare the z - 500 nm propagation to the 0 nm retrieval
    sample_name="Data_for_Jonathon_z0_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    with open(filename,"rb") as file:
        obj=pickle.load(file)
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(obj,"Retrieval at z=0 um")
    # plt.savefig("z0retrieval")
    plt.close(fig)
    retrieved['0'] = obj

    # plot the original retrieval at -500 nm
    sample_name="Data_for_Jonathon_z-500_1-fits_ud_1-0_reconstructed.p"
    filename=os.path.join(folder_dir,sample_name)
    with open(filename,"rb") as file:
        obj=pickle.load(file)
    fig=diffraction_functions.plot_amplitude_phase_meas_retreival(obj,"Retrieval at z=-500 um")
    # plt.savefig("z-500retrieval")
    plt.close(fig)
    retrieved['-500'] = obj

    make_nice_figure(retrieved)

    plt.show()
