import numpy as np
from numpy import unravel_index
import scipy
import diffraction_functions
import matplotlib.pyplot as plt
import diffraction_net
import tables
import pickle
import os
from scipy import interpolate
import argparse

def get_interpolation_points(amplitude_mask):
    """
        get the points for bilinear interp
    """
    x=[]
    y=[]

    # plt.figure()
    # plt.title("Left Points")
    # plt.pcolormesh(amplitude_mask)
    for col in [35,41,48,54,61]:
        for row in [93,86,80,74,67,61,54,48,41,35]:
            x.append(col)
            y.append(row)
            # plt.axvline(x=col,color="red")
            # plt.axhline(y=row,color="blue")

    # plt.figure()
    # plt.title("Right Upper Points")
    # plt.pcolormesh(amplitude_mask)
    for col in [67,73,80,86,93]:
        for row in [93,86,80,74,67]:
            x.append(col)
            y.append(row)
            # plt.axvline(x=col,color="red")
            # plt.axhline(y=row,color="blue")

    # plt.figure()
    # plt.title("Right Lower Points")
    # plt.pcolormesh(amplitude_mask)
    for col in [66,72,79,85,91]:
        for row in [62,55,49,42,36]:
            x.append(col)
            y.append(row)
            # plt.axvline(x=col,color="red")
            # plt.axhline(y=row,color="blue")

    return x,y

class CompareNetworkIterative():
    def __init__(self, args):
        # retrieve image with neural network
        self.network=diffraction_net.DiffractionNet(args.network) # load a pre trained network
        self.args=args

    def test(self,index,folder):
        m_index=(64,64)
        # load diffraction pattern
        # index=11
        # index=9 # best
        N=None
        with tables.open_file("zernike3/build/test_noise.hdf5",mode="r") as file:
        # with tables.open_file("zernike3/build/test.hdf5",mode="r") as file: # use the noise free sample, and matlab result looks good
            N = file.root.N[0,0]
            object_real = file.root.object_real[index, :].reshape(N,N)
            object_imag = file.root.object_imag[index, :].reshape(N,N)
            diffraction = file.root.diffraction_noise[index, :].reshape(N,N)
            diffraction_noisefree = file.root.diffraction_noisefree[index, :].reshape(N,N)

        actual_object = {}
        actual_object["measured_pattern"] = diffraction
        actual_object["tf_reconstructed_diff"] = diffraction_noisefree
        actual_object["real_output"] = object_real
        actual_object["imag_output"] = object_imag

        # fig=diffraction_functions.plot_amplitude_phase_meas_retreival(actual_object,"actual_object",ACTUAL=True,m_index=m_index,mask=False)

        # # get the reconstructed diffraction pattern and the real / imaginary object
        nn_retrieved = {}
        nn_retrieved["measured_pattern"] = diffraction
        nn_retrieved["tf_reconstructed_diff"] = self.network.sess.run(
                self.network.nn_nodes["recons_diffraction_pattern"], feed_dict={self.network.x:diffraction.reshape(1,N,N,1)})
        nn_retrieved["real_output"] = self.network.sess.run(
                self.network.nn_nodes["real_out"], feed_dict={self.network.x:diffraction.reshape(1,N,N,1)})
        nn_retrieved["imag_output"] = self.network.sess.run(
                self.network.nn_nodes["imag_out"], feed_dict={self.network.x:diffraction.reshape(1,N,N,1)})

        # with open("nn_retrieved.p","wb") as file:
            # pickle.dump(nn_retrieved,file)
        # with open("nn_retrieved.p","rb") as file:
            # nn_retrieved=pickle.load(file)

        # plot retrieval with neural network
        # fig=diffraction_functions.plot_amplitude_phase_meas_retreival(nn_retrieved,"nn_retrieved",m_index=m_index)

        # get amplitude mask
        N = np.shape(nn_retrieved["measured_pattern"])[1]
        _, amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(N/2))
        # get interpolation points

        # run matlab retrieval with and without interpolation
        matlabcdi_retrieved_interp=diffraction_functions.matlab_cdi_retrieval(np.squeeze(nn_retrieved['measured_pattern']),amplitude_mask,interpolate=True)
        # with open("matlab_cdi_retrieval.p","wb") as file:
            # pickle.dump(matlabcdi_retrieved_interp,file)
        # with open("matlab_cdi_retrieval.p","rb") as file:
            # matlabcdi_retrieved_interp=pickle.load(file)

        # fig=diffraction_functions.plot_amplitude_phase_meas_retreival(matlabcdi_retrieved_interp,"matlabcdi_retrieved_interp",m_index=m_index)

        # compare and calculate phase + intensity error
        network={}
        iterative={}

        phase_rmse,intensity_rmse=intensity_phase_error(actual_object,matlabcdi_retrieved_interp,"matlabcdi_retrieved_interp_"+str(index),folder)
        print("phase_rmse: ",phase_rmse,"  intensity_rmse: ",intensity_rmse)
        iterative['phase_rmse']=phase_rmse
        iterative['intensity_rmse']=intensity_rmse

        phase_rmse,intensity_rmse=intensity_phase_error(actual_object,nn_retrieved,"nn_retrieved_"+str(index),folder)
        print("phase_rmse: ",phase_rmse,"  intensity_rmse: ",intensity_rmse)
        network['phase_rmse']=phase_rmse
        network['intensity_rmse']=intensity_rmse

        # rmse=intensity_phase_error(actual_object,nn_retrieved)
        # actual_object
        # matlabcdi_retrieved_interp
        # nn_retrieved
        # plt.close('all')
        # plt.show()
        return network,iterative

def intensity_phase_error(actual,predicted,title,folder):
    """
    actual, predicted
    : dictionaries with keys:

    measured_pattern
    tf_reconstructed_diff
    real_output
    imag_output

    """
    actual_c = actual["real_output"]+1j*actual["imag_output"]
    predicted_c = predicted["real_output"]+1j*predicted["imag_output"]
    actual_c=np.squeeze(actual_c)
    predicted_c=np.squeeze(predicted_c)
    # both normalized
    actual_c=actual_c/np.max(np.abs(actual_c))
    predicted_c=predicted_c/np.max(np.abs(predicted_c))

    print("title: ",title)
    print("np.max(np.abs(actual_c)) =>", np.max(np.abs(actual_c)))
    print("np.max(np.abs(predicted_c)) =>", np.max(np.abs(predicted_c)))

    # get wavefront sensor
    N=np.shape(actual_c)[0]
    _,amplitude_mask=diffraction_functions.get_amplitude_mask_and_imagesize(N,int(N/2))
    # get wavefront sensor boundary
    w_l=0
    w_r=N-1
    w_t=0
    w_b=N-1
    while np.sum(amplitude_mask,axis=1)[w_t]==0:
        w_t+=1
    while np.sum(amplitude_mask,axis=1)[w_b]==0:
        w_b-=1
    while np.sum(amplitude_mask,axis=0)[w_l]==0:
        w_l+=1
    while np.sum(amplitude_mask,axis=0)[w_r]==0:
        w_r-=1

    w_t+=8
    w_b-=8
    w_l+=8
    w_r-=8

    # set both to 0 outside wavefront sensor area
    predicted_c[0:w_t,:]=0
    predicted_c[w_b:,:]=0
    predicted_c[:,0:w_l]=0
    predicted_c[:,w_r:]=0
    actual_c[0:w_t,:]=0
    actual_c[w_b:,:]=0
    actual_c[:,0:w_l]=0
    actual_c[:,w_r:]=0

    # # set both to 0 at less than 50% predicted peak
    actual_c[np.abs(actual_c)**2 < 0.05 * np.max(np.abs(actual_c)**2)] = 0.0
    predicted_c[np.abs(actual_c)**2 < 0.05 * np.max(np.abs(actual_c)**2)] = 0.0

    actual_I = np.abs(actual_c)**2
    predicted_I = np.abs(predicted_c)**2

    # find intensity peak of predicted
    m_index = unravel_index(actual_I.argmax(), actual_I.shape)
    # m_index = [int(N/2),int(N/2)]
    # predicted_phase_Imax = np.angle(predicted_c[m_index[0], m_index[1]])
    # actual_phase_Imax = np.angle(actual_c[m_index[0], m_index[1]])

    # phase angle scan to find smallest phase error
    min_phase_angle=None
    min_phase_mse=999.0
    for d_phi in np.linspace(-2*np.pi,2*np.pi,1000):
        _predicted_c=np.array(predicted_c)
        _predicted_c*=np.exp(-1j * d_phi)
        # phase rmse
        A = np.angle(actual_c).reshape(-1)
        B = np.angle(_predicted_c).reshape(-1)
        phase_mse = (np.square(A-B)).mean()
        if phase_mse < min_phase_mse:
            min_phase_mse=phase_mse
            min_phase_angle=d_phi

    # subtract phase at center
    predicted_c *= np.exp(-1j * min_phase_angle)
    # actual_c *= np.exp(-1j * actual_phase_Imax)


    # phase rmse
    A = np.angle(actual_c).reshape(-1)
    B = np.angle(predicted_c).reshape(-1)
    phase_rmse = np.sqrt((np.square(A-B)).mean())

    # intensity mse
    A = actual_I.reshape(-1)
    B = predicted_I.reshape(-1)
    intensity_rmse = np.sqrt((np.square(A-B)).mean())


    fig = plt.figure(figsize=(10,10))
    fig.suptitle(title)
    gs = fig.add_gridspec(4,2)

    # plot rmse
    fig.text(0.2, 0.95, "intensity_rmse:"+str(intensity_rmse)+"\n"+"  phase_rmse"+str(phase_rmse)
            , ha="center", size=12, backgroundcolor="cyan")

    # measured diffraction pattern
    ax=fig.add_subplot(gs[0,0])
    ax.set_title("Measured Diffraction Pattern")
    im=ax.pcolormesh(np.squeeze(predicted['measured_pattern']))
    fig.colorbar(im,ax=ax)

    ax=fig.add_subplot(gs[0,1])
    ax.set_title("Reconstructed Diffraction Pattern")
    im=ax.pcolormesh(np.squeeze(predicted['tf_reconstructed_diff']))
    fig.colorbar(im,ax=ax)

    # intensity
    ax=fig.add_subplot(gs[1,0])
    ax.set_title("actual_I")
    im=ax.pcolormesh(actual_I)
    ax.axvline(x=m_index[1],color="red",alpha=0.8)
    ax.axhline(y=m_index[0],color="blue",alpha=0.8)
    fig.colorbar(im,ax=ax)

    ax=fig.add_subplot(gs[1,1])
    ax.set_title("predicted_I")
    im=ax.pcolormesh(predicted_I)
    ax.axvline(x=m_index[1],color="red",alpha=0.8)
    ax.axhline(y=m_index[0],color="blue",alpha=0.8)
    fig.colorbar(im,ax=ax)

    ax=fig.add_subplot(gs[2,0])
    ax.set_title("actual_c angle")
    im=ax.pcolormesh(np.angle(actual_c))
    ax.axvline(x=m_index[1],color="red",alpha=0.8)
    ax.axhline(y=m_index[0],color="blue",alpha=0.8)
    fig.colorbar(im,ax=ax)

    ax=fig.add_subplot(gs[3,0])
    ax.set_title("actual_c angle")
    ax.plot(np.angle(actual_c)[m_index[0],:])
    ax.axvline(x=m_index[1],color="red")

    ax=fig.add_subplot(gs[2,1])
    ax.set_title("predicted_c angle")
    im=ax.pcolormesh(np.angle(predicted_c))
    ax.axvline(x=m_index[1],color="red",alpha=0.8)
    ax.axhline(y=m_index[0],color="blue",alpha=0.8)
    fig.colorbar(im,ax=ax)

    ax=fig.add_subplot(gs[3,1])
    ax.set_title("predicted_c angle")
    ax.plot(np.angle(predicted_c)[m_index[0],:])
    ax.axvline(x=m_index[1],color="red")

    if not os.path.isdir(folder):
        os.mkdir(folder)
    fig.savefig(os.path.join(folder,title))

    # plt.figure(105)
    # plt.pcolormesh(np.angle(predicted_c) - np.angle(actual_c))
    # plt.gca().axvline(x=m_index[1],color="red",alpha=0.8)
    # plt.gca().axhline(y=m_index[0],color="blue",alpha=0.8)
    # plt.colorbar()

    return phase_rmse,intensity_rmse




# class contains list of error values,
class ErrorDistribution():
    def __init__(self):
        self.values=[]
        self.standard_deviation=None
        self.average=None
    def calculate_statistics(self):
        self.standard_deviation=np.std(self.values)
        self.average=np.average(self.values)


class PhaseIntensityError():
    def __init__(self):
        self.phase_error=ErrorDistribution()
        self.intensity_error=ErrorDistribution()

    def calculate_statistics(self):
        self.phase_error.calculate_statistics()
        self.intensity_error.calculate_statistics()


if __name__ == "__main__":

    # TODO : evaluate rmse at high intensity areas
    # + phase, set constant phase shift

    # evaluate at different noise levels

    # run a variational network, run an RNN network

    parser=argparse.ArgumentParser()
    parser.add_argument('--network',type=str)
    parser.add_argument('--pc',type=str)
    args=parser.parse_args()
    comparenetworkiterative = CompareNetworkIterative(args)

    network_error_phase=[]
    network_error_intensity=[]
    iterative_error_phase=[]
    iterative_error_intensity=[]

    network_error=PhaseIntensityError()
    iterative_error=PhaseIntensityError()
    N = 100
    for i in range(0,N):
        network,iterative=comparenetworkiterative.test(i,'test_pc_'+args.pc)
        network_error.phase_error.values.append(network['phase_rmse'])
        network_error.intensity_error.values.append(network['intensity_rmse'])

        iterative_error.phase_error.values.append(iterative['phase_rmse'])
        iterative_error.intensity_error.values.append(iterative['intensity_rmse'])

    # calculate statistics
    network_error.calculate_statistics()
    iterative_error.calculate_statistics()


    # save these to a pickle
    errorvals={}
    errorvals["network_error"]=network_error
    errorvals["iterative_error"]=iterative_error
    with open('error_'+args.pc+'.p','wb') as file:
        pickle.dump(errorvals,file)


