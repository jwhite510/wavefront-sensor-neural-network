import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    for i in range(9):
        i+=1
        arr=np.loadtxt("opencvm1_complex_after"+str(i)+".dat")
        plt.figure()
        plt.title("shift:"+str((i-1)*2.5)+" pixels")
        plt.gca().set_xlim(150,350)
        plt.gca().set_ylim(60,190)
        plt.imshow(arr)

        plt.gcf().savefig("img"+str(i)+".png")
        # plt.show()

