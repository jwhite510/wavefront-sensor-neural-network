import ctypes

lib=ctypes.cdll.LoadLibrary('libMeasuredImageFormatterWrapper.so')
a=lib.MeasuredImageFormatter_new(30)
lib.MeasuredImageFormatter_PrintInt(a)
