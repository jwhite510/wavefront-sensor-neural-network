import pickle
import numpy as np
import matplotlib.pyplot as plt

def printvals(filename):
    print("     ")
    print("opening file: ", filename)
    with open(filename, "rb") as file:
        obj=pickle.load(file)
    print("np.shape(obj)", np.shape(obj))

    minv_avg,maxv_avg,avg_avg=0,0,0
    for i in range(10):
        obj_s=obj[i,:,:,:]
        # if i == 0:
            # print("np.min(obj_s) =>", np.min(obj_s))
            # print("np.max(obj_s) =>", np.max(obj_s))
        minv_avg+=np.min(obj_s)
        maxv_avg+=np.max(obj_s)
        avg_avg+=np.average(obj_s)
    return minv_avg/10, maxv_avg/10, avg_avg/10


if __name__ == "__main__":

    leakyrelu7_minvals = []
    leakyrelu7_maxvals = []
    leakyrelu7_avgvals = []

    batch_norm8_minvals = []
    batch_norm8_maxvals = []
    batch_norm8_avgvals = []

    for s in range(1,32):
        minv,maxv,avgv=printvals("leakyrelu7_t__"+str(s)+".p")
        leakyrelu7_minvals.append(minv)
        leakyrelu7_maxvals.append(maxv)
        leakyrelu7_avgvals.append(avgv)

        minv,maxv,avgv=printvals("batch_norm8_t__"+str(s)+".p")
        batch_norm8_minvals.append(minv)
        batch_norm8_maxvals.append(maxv)
        batch_norm8_avgvals.append(avgv)

    plt.figure(2)
    plt.plot(leakyrelu7_minvals,label="leakyrelu7_minvals")
    plt.plot(leakyrelu7_maxvals,label="leakyrelu7_maxvals")
    plt.plot(leakyrelu7_avgvals,label="leakyrelu7_avgvals")

    plt.plot(batch_norm8_minvals,label="batch_norm8_minvals")
    plt.plot(batch_norm8_maxvals,label="batch_norm8_maxvals")
    plt.plot(batch_norm8_avgvals,label="batch_norm8_avgvals")

    plt.gca().legend()
    plt.show()


