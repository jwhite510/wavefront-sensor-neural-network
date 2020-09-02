from PIL import Image, ImageDraw
import PIL.ImageOps
import PIL
import numpy as np


class MaterialParams():
  def __init__(self,beta_Ta,delta_Ta,distance,dz,lam):
      self.beta_Ta=beta_Ta
      self.delta_Ta=delta_Ta
      self.distance=distance
      self.dz=dz
      self.lam=lam
      self.k=2*np.pi/lam

class Parameters():
    def __init__(self):
        self.wavefront_sensor=None
        self.wavefron_sensor_size_nm=None
        self.material_params=[]

params = Parameters()

# wavefront sensor as PIL image

im = Image.open("size_6um_pitch_600nm_diameter_300nm_psize_5nm.png")
params.wavefront_sensor=im
params.wavefron_sensor_size_nm=5*im.size[0]*1e-9

# # for the 6x6 image in same format as the 600nm
# # im2 = Image.open("10x10.png")
# im2 = Image.open("6x6.png")
# im2=im2.resize((1200,1200))
# im3=np.zeros((1200,1200,3),dtype=np.array(im2).dtype)
# im3[:,:,0]=np.array(im2)[:,:,3]
# im3[:,:,1]=np.array(im2)[:,:,3]
# im3[:,:,2]=np.array(im2)[:,:,3]
# im3=Image.fromarray(im3)
# im3=PIL.ImageOps.invert(im3)
# im=im3
# params.wavefront_sensor=im
# params.wavefron_sensor_size_nm=60e-6

# define materials
# https://refractiveindex.info/?shelf=main&book=Cu&page=Johnson
# https://refractiveindex.info/?shelf=main&book=Si3N4&page=Luke
# wavefront sensor material properties
# params_cu = MaterialParams(
    # lam=633e-9,
    # dz=10e-9,
    # delta_Ta = 0.26965-1, # double check this
    # beta_Ta = 3.4106,
    # distance = 150e-9
        # )
# params.material_params.append(params_cu)

# params_Si = MaterialParams(
    # delta_Ta = 2.0394-1,
    # beta_Ta = 0.0,
    # dz=10e-9,
    # lam=633e-9,
    # distance=50e-9
        # )
# params.material_params.append(params_Si)

# XUV 
params_Si = MaterialParams(
    beta_Ta = 0.00926,
    delta_Ta = 0.02661,
    dz=10e-9,
    lam=18.5e-9,
    distance=50e-9)

params.material_params.append(params_Si)
params_cu = MaterialParams(
    beta_Ta = 0.0612,
    delta_Ta = 0.03748,
    lam=18.5e-9,
    dz=10e-9,
    distance = 150e-9)
params.material_params.append(params_cu)


params.wf_ratio=1/2
