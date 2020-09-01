import numpy as np
import matplotlib.pyplot as plt
import datagen
import pickle
from find_similar_diffraction import Sample

if __name__=="__main__":

    with open("sample1.p","rb") as file:
        sample1=pickle.load(file)
    with open("sample2.p","rb") as file:
        sample2=pickle.load(file)

    print(sample1.index,sample2.index)
    datagenerator = datagen.DataGenerator(1024,128)
    datagen.plotsamples(sample1.beforewf,sample1.afterwf,sample1.z_coefs,sample1.scales,datagenerator.zernike_cvector,'test')
    datagen.plotsamples(sample2.beforewf,sample2.afterwf,sample2.z_coefs,sample2.scales,datagenerator.zernike_cvector,'test')
    plt.show()
