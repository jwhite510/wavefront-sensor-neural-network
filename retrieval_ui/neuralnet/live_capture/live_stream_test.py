import cv2
import numpy as np
import TIS
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Open camera, set video format, framerate and determine, whether the sink is color or bw
    # Parameters: Serialnumber, width, height, framerate (numerator only) , color
    # If color is False, then monochrome / bw format is in memory. If color is True, then RGB32
    # colorformat is in memory
    Tis = TIS.TIS("48710182", 640, 480, 30, False)
    Tis.Start_pipeline()  # Start the pipeline so the camera streams

    print('Press Esc to stop')
    lastkey = 0

    cv2.namedWindow('Window')  # Create an OpenCV output window

    kernel = np.ones((5, 5), np.uint8)  # Create a Kernel for OpenCV erode function

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    plt.ion()
    first_image = False

    while lastkey != 27:
            if Tis.Snap_image(1) is True:  # Snap an image with one second timeout
                    image = Tis.Get_image()  # Get the image. It is a numpy array

                    if not first_image:
                        im = ax.imshow(image[:,:,0], vmin=0.0, vmax=20)
                        colorbar = fig.colorbar(im, ax=ax)
                        first_image = True
                    else:
                        im.set_data(image[:,:,0])
                        print("calling set_clim")
                        print("np.min(image) =>", np.min(image))
                        print("np.max(image) =>", np.max(image))
                        im.set_clim(vmin=np.min(image), vmax=np.max(image))

                    # print("np.min(image) =>", np.min(image))
                    # print("np.max(image) =>", np.max(image))

                    plt.pause(0.001)

                    image = cv2.erode(image, kernel, iterations=5)  # Example OpenCV image processing
                    cv2.imshow('Window', image)  # Display the result

            lastkey = cv2.waitKey(10)

    # Stop the pipeline and clean up
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()
    print('Program ends')



