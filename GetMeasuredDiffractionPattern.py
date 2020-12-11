import numpy as np
import matplotlib.pyplot as plt
import diffraction_functions
from  astropy.io import fits
import params
import time

class GetMeasuredDiffractionPattern():
    """
    for getting a measured diffraction pattern and formatting it to match the
    frequency axis of the simulation grid
    """
    def __init__(self, N_sim, N_meas, experimental_params):
        """
        experimental_params: dict{}
        experimental_params['pixel_size']
        experimental_params['z_distance']
        experimental_params['wavelength']
        """


        # get the axes of the simulation
        self.N_sim = N_sim # size of the measured simulated pattern
        self.experimental_params = experimental_params
        self.N_meas = N_meas
        self.simulation_axes, _ = diffraction_functions.get_amplitude_mask_and_imagesize(self.N_sim, int(params.params.wf_ratio*N_sim))

        # parameters for the input measured diffraction pattern
        self.measured_axes = {}
        self.measured_axes["diffraction_plane"]={}
        self.calculate_measured_axes()

        # calculate ratio of df
        self.df_ratio = self.measured_axes['diffraction_plane']['df'] / self.simulation_axes['diffraction_plane']['df']
        # multiply by scale adjustment

        # multiply by scale close to 1 (to account for errors)
        # self.df_ratio *= # fine adjustment

    def calculate_measured_axes(self):
        # calculate delta frequency
        self.measured_axes["diffraction_plane"]["df"] = self.experimental_params['pixel_size'] / (self.experimental_params['wavelength'] * self.experimental_params['z_distance'])
        self.measured_axes["diffraction_plane"]["xmax"] = self.N_meas * (self.experimental_params['pixel_size'] / 2)
        self.measured_axes["diffraction_plane"]["x"] = np.arange(-(self.measured_axes["diffraction_plane"]["xmax"]), (self.measured_axes["diffraction_plane"]["xmax"]), self.experimental_params['pixel_size'])
        self.measured_axes["diffraction_plane"]["f"] = self.measured_axes["diffraction_plane"]["x"] / (self.experimental_params['wavelength'] * self.experimental_params['z_distance'])
        # self.measured_axes["diffraction_plane"]["df"] = self.measured_axes["diffraction_plane"]["f"][-1] - self.measured_axes["diffraction_plane"]["f"][-2]


    def format_measured_diffraction_pattern(self, measured_diffraction_pattern, transform):
        """
        measured_diffraction_pattern (numpy array)
        transform["rotation_angle"] = float/int (degrees)
        transform["scale"] = integer
        transform["flip"] = "lr" "ud" "lrud" None
        """
        print('BBBBBBBBBBBBBBBB')
        print('format_measured_diffraction_pattern running')
        time_a=time.time()

        measured_diffraction_pattern = diffraction_functions.format_experimental_trace(
            N=self.N_sim,
            df_ratio=self.df_ratio*transform["scale"],
            measured_diffraction_pattern=measured_diffraction_pattern,
            rotation_angle=transform["rotation_angle"],
            trim=1) # if transposed (measured_pattern.T) , flip the rotation

        time_b=time.time()
        print('format_experimental_trace total time:',time_b-time_a)
        time_a=time.time()


        if transform["flip"] is not None:
            if transform["flip"] == "lr":
                measured_diffraction_pattern = np.flip(measured_diffraction_pattern, axis=1)

            elif transform["flip"] == "ud":
                measured_diffraction_pattern = np.flip(measured_diffraction_pattern, axis=0)

            elif transform["flip"] == "lrud":
                measured_diffraction_pattern = np.flip(measured_diffraction_pattern, axis=0)
                measured_diffraction_pattern = np.flip(measured_diffraction_pattern, axis=1)
            else:
                raise ValueError("invalid flip specified")
            # center it again

        measured_diffraction_pattern = diffraction_functions.center_image_at_centroid(measured_diffraction_pattern)
        measured_diffraction_pattern = np.expand_dims(measured_diffraction_pattern, axis=0)
        measured_diffraction_pattern = np.expand_dims(measured_diffraction_pattern, axis=-1)
        measured_diffraction_pattern *= (1/np.max(measured_diffraction_pattern))
        time_b=time.time()
        print('center_image_at_centroid:',time_b-time_a)
        return measured_diffraction_pattern


if __name__ == "__main__":

    fits_file_name = "m3_scan_0000.fits"
    thing = fits.open(fits_file_name)
    measured_pattern = thing[0].data[0,:,:]
    N_meas = np.shape(measured_pattern)[0]
    measured_pattern = measured_pattern.astype(np.float64)

    experimental_params = {}
    experimental_params['pixel_size'] = 27e-6 # [meters] with 2x2 binning
    experimental_params['z_distance'] = 33e-3 # [meters] distance from camera
    experimental_params['wavelength'] = 13.5e-9 #[meters] wavelength

    getMeasuredDiffractionPattern = GetMeasuredDiffractionPattern(N_sim=128, N_meas=N_meas, experimental_params=experimental_params)

    transform={}
    transform["rotation_angle"]=3
    transform["scale"]=1
    # transform["flip"]="lr"
    transform["flip"]=None
    m = getMeasuredDiffractionPattern.format_measured_diffraction_pattern(measured_pattern, transform)
    m2 = diffraction_functions.get_and_format_experimental_trace(128, transform)

    print("np.shape(m) => ",np.shape(m))
    print("np.shape(m2) => ",np.shape(m2))

    plt.figure()
    plt.imshow(np.squeeze(m))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.squeeze(m2))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.squeeze(m)-np.squeeze(m2))
    plt.colorbar()

    plt.show()


