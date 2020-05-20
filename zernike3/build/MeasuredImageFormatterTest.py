import ctypes
import numpy as np

lib=ctypes.cdll.LoadLibrary('libMeasuredImageFormatterWrapper.so')

dif_in=np.ones((4,4),dtype=np.double)
value=0
for i in range(0,4):
    for j in range(0,4):
        dif_in[i,j]=value
        value+=1
print(dif_in)
dif_out=np.ones((2,2),dtype=np.double)

a=lib.MeasuredImageFormatter_new(30,dif_in.ctypes.data,dif_in.shape[0],dif_in.shape[1],
        dif_out.ctypes.data, dif_out.shape[0], dif_out.shape[1])
lib.MeasuredImageFormatter_PrintInt(a)
