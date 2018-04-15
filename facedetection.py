# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
from keras.models import load_model 
 
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where going to be save")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
img = args["template"]
img_rows, img_cols = 24, 24


model = load_model('model.h5')
model.load_weights('weights.h5')

# load the image image, convert it to grayscale
template = cv2.imread(img)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
(tH, tW) = template.shape[:2] # Get image size

delta = np.linspace(1, 0.05, 100, endpoint=True)

for scale in delta:
    # Resize the image according to the scale, and keep track of the ratio of the resizing 
    resized = imutils.resize(template, width = int(template.shape[1] * scale))
    r = template.shape[1] / float(resized.shape[1])
    
    aux = []
    aux2 = []
    for y in range(template.shape[0]):
        for x in range(template.shape[1]):
            crop_img = resized[y:y+24, x:x+24]
            aux.append(crop_img)
            crop_img = crop_img.reshape(x.shape[0], img_rows, img_cols, 1)
            aux2.append(model.predict(crop_img))
            
    
    if resized.shape[0] < 24 or resized.shape[1] < 24:
        break