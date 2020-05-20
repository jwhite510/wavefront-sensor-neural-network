import ctypes
import numpy as np

lib=ctypes.cdll.LoadLibrary('libMeasuredImageFormatterWrapper.so')

# dif_in=np.zeros((4,4),dtype=np.double)
# dif_in=np.zeros((256,256),dtype=np.double)
dif_in=np.zeros((512,512),dtype=np.double)
dif_in[50:100,30:80]=1
c_double_p=ctypes.POINTER(ctypes.c_double)
dif_out=np.zeros((256,256),dtype=np.double)
a=lib.MeasuredImageFormatter_new(
        ctypes.c_double(1.5),
        dif_in.ctypes.data_as(c_double_p),dif_in.shape[0],dif_in.shape[1],
        dif_out.ctypes.data_as(c_double_p), dif_out.shape[0], dif_out.shape[1]
        )

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
