# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
 
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where going to be save")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
img = args["template"]

# load the image image, convert it to grayscale
template = cv2.imread(img)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
(tH, tW) = template.shape[:2] # Get image size

delta = np.linspace(1, 0.05, 100, endpoint=True)

for scale in delta:
    # Resize the image according to the scale, and keep track of the ratio of the resizing 
    resized = imutils.resize(template, width = int(template.shape[1] * scale))
    r = template.shape[1] / float(resized.shape[1])
    # cv2.imshow("Template", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()   
    # if the resized image is smaller than 24x24, then break from the loop
    if resized.shape[0] < 24 or resized.shape[1] < 24:
        break