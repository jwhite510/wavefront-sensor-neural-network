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
    for _name, _i, _dist in zip(['-500','0','300_sim'],range(3),[-500,0,300]):
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
            ax.set_xlabel(r"positon [um]")
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
            ax.set_xlabel(r"positon [um]")
        ax.text(1.3,0.5,'z='+str(_dist)+' [um]',transform=ax.transAxes,size=20)
    fig.savefig('xuv_experimental_results_1.png')

    fig=plt.figure(figsize=(10,8))
    fig.subplots_adjust(left=0.0,right=1.0)
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    im = Image.open('figure_images/xuv_experimental_setup.png')
    im = np.array(im)
    ax.imshow(im)
    ax.text(0.1,0.0,'z=300 [um]',transform=ax.transAxes,size=20,ha='left')
    ax.text(0.45,0.0,'z=0 [um]',transform=ax.transAxes,size=20,ha='left')
    ax.text(0.7,0.0,'z=-500 [um]',transform=ax.transAxes,size=20,ha='left')
    ax.axis('off')
    fig.savefig('xuv_experimental_results_2.png')

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
    z = 800e-6
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
    retrieved['300_sim'] = {
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
