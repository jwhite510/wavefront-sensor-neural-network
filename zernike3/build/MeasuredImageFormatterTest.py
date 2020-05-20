import ctypes
import numpy as np

lib=ctypes.cdll.LoadLibrary('libMeasuredImageFormatterWrapper.so')

dif_in=np.ones((2,2),dtype=np.double)
dif_out=np.ones((2,2),dtype=np.double)
a=lib.MeasuredImageFormatter_new(30,dif_in.ctypes.data,dif_in.shape[0],dif_in.shape[1],
        dif_out.ctypes.data, dif_out.shape[0], dif_out.shape[1])

print(dif_in)
print(dif_out)
# lib.MeasuredImageFormatter_PrintInt(a)
lib.MeasuredImageFormatter_Format(a)
print(dif_in)
print(dif_out)


print("-----------------")
dif_in*=2
print(dif_in)
print(dif_out)
# lib.MeasuredImageFormatter_PrintInt(a)
lib.MeasuredImageFormatter_Format(a)
print(dif_in)
print(dif_out)
