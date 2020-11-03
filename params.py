from PIL import Image, ImageDraw
import PIL.ImageOps
import PIL
import numpy as np
import argparse

def custom_round(coordinates:list,radius:float,size:tuple)->np.array:
    image = 255*np.ones((size[0],size[1],3),dtype=np.uint8)
    row=np.arange(0,size[0]).reshape(-1,1)
    col=np.arange(0,size[1]).reshape(1,-1)
    for _c in coordinates:
        # radius
        # draw distance
        dist = np.sqrt((row-_c[0])**2 + (col-_c[1])**2)
        image[dist<radius,:]=0

    return Image.fromarray(image)


def wfs_square_6x6_upright():
    # for the 6x6 image in same format as the 600nm
    im2 = Image.open("6x6.png")
    im2=im2.resize((1200,1200))
    im3=np.zeros((1200,1200,3),dtype=np.array(im2).dtype)
    im3[:,:,0]=np.array(im2)[:,:,3]
    im3[:,:,1]=np.array(im2)[:,:,3]
    im3[:,:,2]=np.array(im2)[:,:,3]
    im3=Image.fromarray(im3)
    im3=PIL.ImageOps.invert(im3)
    im=im3
    return im

def wfs_square_10x10_upright():
    # for the 6x6 image in same format as the 600nm
    im2 = Image.open("10x10.png")
    im2=im2.resize((1200,1200))
    im3=np.zeros((1200,1200,3),dtype=np.array(im2).dtype)
    im3[:,:,0]=np.array(im2)[:,:,3]
    im3[:,:,1]=np.array(im2)[:,:,3]
    im3[:,:,2]=np.array(im2)[:,:,3]
    im3=Image.fromarray(im3)
    im3=PIL.ImageOps.invert(im3)
    im=im3
    return im

def wfs_round_10x10_downleft():
    return Image.open("size_6um_pitch_600nm_diameter_300nm_psize_5nm.png")

def wfs_round_10x10_upright():
    return Image.open("size_6um_pitch_600nm_diameter_300nm_psize_5nm_up_left.png")

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

parser=argparse.ArgumentParser()
parser.add_argument('--wfsensor',type=int)
args,_=parser.parse_known_args()
if args.wfsensor==None: raise ValueError('wfs argument not passed')

# # used in XUV experiment
if args.wfsensor==0:
    params.wavefront_sensor=wfs_round_10x10_downleft()

elif args.wfsensor==1:
    # make square shape
    coordinates,side_offset,size= [],60,(1200,1200)
    np.random.seed(5678)
    random_offset_scale=40.0 # set this to and it will be exactly the same as 10x10 downleft wfs
    for _r in np.linspace(0+side_offset,size[0]-side_offset,10):
        for _c in np.linspace(0+side_offset,size[1]-side_offset,10):
            random_r_offset=(np.random.rand()-0.5)*random_offset_scale
            random_c_offset=(np.random.rand()-0.5)*random_offset_scale
            if _r < size[0]//2 and _c > size[1]//2:
                coordinates.append((_r+25+random_r_offset,_c-25+random_c_offset))
            else: coordinates.append((_r+random_r_offset,_c+random_c_offset))
    params.wavefront_sensor=custom_round(
            coordinates,
            radius=30,
            size=size
            )

elif args.wfsensor==2:
    # used in visible light experiment
    params.wavefront_sensor=wfs_square_6x6_upright()

elif args.wfsensor==3:
    # the modified wavefront sensor that matches the up / right part of the square ones
    params.wavefront_sensor=wfs_round_10x10_upright()

elif args.wfsensor==4:
    params.wavefront_sensor=wfs_square_10x10_upright()

# size used in XUV
params.wavefron_sensor_size_nm=5*params.wavefront_sensor.size[0]*1e-9

# size used in visible
# params.wavefron_sensor_size_nm=60e-6

# define materials
# https://refractiveindex.info/?shelf=main&book=Cu&page=Johnson
# https://refractiveindex.info/?shelf=main&book=Si3N4&page=Luke
# wavefront sensor material properties
# params_cu = MaterialParams(
#     lam=633e-9,
#     dz=10e-9,
#     delta_Ta = 0.26965-1, # double check this
#     beta_Ta = 3.4106,
#     distance = 150e-9
#         )
# params.material_params.append(params_cu)
# 
# params_Si = MaterialParams(
#     delta_Ta = 2.0394-1,
#     beta_Ta = 0.0,
#     dz=10e-9,
#     lam=633e-9,
#     distance=50e-9
#         )
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
