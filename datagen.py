import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def makezernike(m: int,n: int, N_computational: int)->np.array:
    x = np.linspace(-7,7,N_computational).reshape(-1,1)
    y = np.linspace(-7,7,N_computational).reshape(1,-1)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x,y)

    positive_m = False
    if m >= 0:
        positive_m=True
    m = abs(m)
    zernike_polynom = np.zeros((N_computational,N_computational))
    k=0
    while k <= (n-m)/2:

        numerator = -1 ** k
        numerator *= math.factorial(n-k)

        denominator = math.factorial(k)
        denominator *= math.factorial(((n+m)/2)-k)
        denominator *= math.factorial(((n-m)/2)-k)

        scalar = numerator/denominator

        zernike_polynom+=scalar*rho**(n-2*k)
        k+=1

    if positive_m:
        zernike_polynom *= np.cos(m * phi)
    else:
        zernike_polynom *= np.sin(m * phi)

    # set values outside unit circle to 0
    zernike_polynom[rho>1]=0
    return zernike_polynom

class Zernike_C():
    def __init__(self,m:int,n:int):
        self.m=m
        self.n=n

class DataGenerator():
    def __init__(self,N_computational:int,N_interp:int,crop_size:int):
        # generate zernike coefficients
        start_n=2
        max_n=4
        zernike_cvector = []

        for n in range(start_n,max_n+1):
            for m in range(n,-n-2,-2):
                zernike_cvector.append(Zernike_C(m,n))

        mn_polynomials=np.zeros((len(zernike_cvector),N_computational,N_computational))
        mn_polynomials_index=0
        for _z in zernike_cvector:
            z_arr = makezernike(_z.m,_z.n,N_computational)
            mn_polynomials[mn_polynomials_index,:,:]=z_arr
            mn_polynomials_index+=1


if __name__ == "__main__":
    datagenerator = DataGenerator(1024,128,200)
    pass

    # initialize tensorflow data generation

