import matplotlib.pyplot as plt
import ctypes
import numpy as np
import diffraction_functions

lib=ctypes.cdll.LoadLibrary('libMeasuredImageFormatterWrapper.so')

# dif_in=np.zeros((4,4),dtype=np.double)
# dif_in=np.zeros((256,256),dtype=np.double)

s=diffraction_functions.fits_to_numpy("../../m3_scan_0000.fits")

# df ratio
df_ratio=0.7272727272727273
rot_angle=-3


dif_in=np.zeros(np.shape(s),dtype=np.double)
np.copyto(dif_in,s)
c_double_p=ctypes.POINTER(ctypes.c_double)
dif_out=np.zeros((128,128),dtype=np.double)
a=lib.MeasuredImageFormatter_new(
        ctypes.c_double(df_ratio),ctypes.c_double(rot_angle),
        dif_in.ctypes.data_as(c_double_p),dif_in.shape[0],dif_in.shape[1],
        dif_out.ctypes.data_as(c_double_p), dif_out.shape[0], dif_out.shape[1]
        )

print(dif_in)
lib.MeasuredImageFormatter_Format(a)
exit()


print("-----------------")
dif_in*=2
print(dif_in)
print(dif_out)
# lib.MeasuredImageFormatter_PrintInt(a)
lib.MeasuredImageFormatter_Format(a)
print(dif_in)
print(dif_out)
