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

    while lastkey != 27:
            if Tis.Snap_image(1) is True:  # Snap an image with one second timeout
                    image = Tis.Get_image()  # Get the image. It is a numpy array
                    image = cv2.erode(image, kernel, iterations=5)  # Example OpenCV image processing
                    print("np.shape(image) =>", np.shape(image))
                    cv2.imshow('Window', image)  # Display the result

            lastkey = cv2.waitKey(10)

    # Stop the pipeline and clean up
    Tis.Stop_pipeline()
    cv2.destroyAllWindows()
    print('Program ends')



