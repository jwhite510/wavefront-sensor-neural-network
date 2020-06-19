import ctypes
import numpy as np

if __name__ == "__main__":
    lib=ctypes.cdll.LoadLibrary('libzernikecuda.so')
    a=lib.zcuda_new()

