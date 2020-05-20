import ctypes
import numpy as np

lib=ctypes.cdll.LoadLibrary('libMeasuredImageFormatterWrapper.so')

dif_in=np.zeros((8,8),dtype=np.double)
dif_in[1:3,1:3]=1
dif_out=np.zeros((16,16),dtype=np.double)
a=lib.MeasuredImageFormatter_new(ctypes.c_double(2.0),dif_in.ctypes.data,dif_in.shape[0],dif_in.shape[1],
        dif_out.ctypes.data, dif_out.shape[0], dif_out.shape[1])

print(dif_in)
# print(dif_out)
# lib.MeasuredImageFormatter_PrintInt(a)
lib.MeasuredImageFormatter_Format(a)
# print(dif_in)
# print(dif_out)
exit()


print("-----------------")
dif_in*=2
print(dif_in)
print(dif_out)
# lib.MeasuredImageFormatter_PrintInt(a)
lib.MeasuredImageFormatter_Format(a)
print(dif_in)
print(dif_out)
