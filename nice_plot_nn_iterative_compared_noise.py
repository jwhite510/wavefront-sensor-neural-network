import argparse
import matplotlib.pyplot as plt
import datagen
import numpy as np
import pickle
import diffraction_functions
import params

def place_colorbar(im,ax,offsetx,offsety,ticks:list,color:str,labels:list=None):
    caxis=fig.add_axes([ax.get_position().bounds[0]+offsetx,ax.get_position().bounds[1]+offsety,0.07,0.01])
    fig.colorbar(im,cax=caxis,orientation='horizontal')
    caxis.xaxis.set_ticks_position('top')
    caxis.xaxis.set_ticks(ticks)
    if labels: caxis.xaxis.set_ticklabels(labels)
    caxis.tick_params(axis='x',colors=color)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--wfsensor',type=int)
    args,_=parser.parse_known_args()
    files={}
    cts = ['0','50','40','30','20','10','5']
    for filename in ['out_'+_ct+'.p' for _ct in cts]:
        with open(filename,'rb') as file:
            files[filename.split('.')[0]]=pickle.load(file)

    N=np.shape(np.squeeze(files['out_0']['retrieved_nn']['measured_pattern']))[0]
    simulation_axes, _ = diffraction_functions.get_amplitude_mask_and_imagesize(N, int(params.params.wf_ratio*N))
    x=simulation_axes['object']['x'] # meters
    x*=1e6
    f=simulation_axes['diffraction_plane']['f'] # 1/meters
    f*=1e-6

    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(wspace=0.00,hspace=0.0)
    # fig.suptitle('nn iterative compared')
    gs = fig.add_gridspec(6,4)
    letter='a'
    for _ct,_z in zip(['0','10'],[0,3]):
        ax=fig.add_subplot(gs[0+_z:2+_z,0:2])
        ax.text(0.05,0.9,letter,transform=ax.transAxes,weight='bold',color='white');letter=chr(ord(letter)+1)
        ax.pcolormesh(f,f,files['out_'+_ct]['retrieved_nn']['measured_pattern'],cmap='jet')
        ax.xaxis.set_ticks([])
        ax.set_title('Simulated Diffraction pattern no noise'if _ct=='0'else'Simulated diffraction pattern 10 counts peak')
        ax.set_ylabel(r"frequency [1/m]$\cdot 10^{6}$")
        for i,retrieval,name in zip(
                [0,1,2],['retrieved_nn','retrieved_iterative','actual'],
                ['Neural Network\nRetrieved ','Iterative\nRetrieved ','Actual ']
                ):
            i+=_z
            ax=fig.add_subplot(gs[i,2])
            ax.text(0.9,0.9,letter,transform=ax.transAxes,weight='bold',color='white');letter=chr(ord(letter)+1)
            complex_obj=np.squeeze(files['out_'+_ct][retrieval]['real_output'] + 1j*files['out_'+_ct][retrieval]['imag_output'])
            complex_obj*=(1/np.max(np.abs(complex_obj)))
            # phase at center set to 0
            complex_obj*=np.exp(-1j * np.angle(complex_obj[N//2,N//2]))

            phase = np.angle(complex_obj)
            phase[np.abs(complex_obj)**2 < 0.005]=0
            im=ax.pcolormesh(x,x,np.abs(complex_obj)**2,vmin=0,vmax=1,cmap='jet')
            ax.text(0.05,0.95,name,transform=ax.transAxes,color='white',weight='bold',va='top')
            # ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
            ax.yaxis.set_ticks([])
            if i==0:ax.set_title('Intensity')
            if not i==5: ax.xaxis.set_ticks([])
            else:ax.set_xlabel(("position [um]"))
            place_colorbar(im,ax,offsetx=0.015,offsety=0.005,ticks=[0,0.5,1],color='white')

            ax=fig.add_subplot(gs[i,3])
            ax.text(0.9,0.9,letter,transform=ax.transAxes,weight='bold',color='black');letter=chr(ord(letter)+1)
            im=ax.pcolormesh(x,x,phase,vmin=-np.pi,vmax=np.pi,cmap='jet')
            ax.text(0.05,0.95,name,transform=ax.transAxes,color='black',weight='bold',va='top')
            # ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
            ax.yaxis.set_ticks([])
            if i==0:ax.set_title('Phase')
            if not i==5: ax.xaxis.set_ticks([])
            else:ax.set_xlabel(("position [um]"))
            place_colorbar(im,ax,offsetx=0.015,offsety=0.005,ticks=[-3.14,0,3.14],color='black',labels=['-pi','0','+pi'])

    # fig.savefig('./wfs_'+str(args.wfsensor)+'beta_98_INTERP_iterative_nn_compared_noise_'+_ct+'.png')
    fig.savefig('./wfs_'+str(args.wfsensor)+'_iterative_nn_compared_noise_'+'multiple'+'.png')



