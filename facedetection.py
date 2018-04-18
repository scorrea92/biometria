# import the necessary packages
%clear
print("Ejecutandose......")
import numpy as np
import argparse
import cv2
import imutils
from keras.models import load_model 
from sklearn import preprocessing
 
# Construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help="Path to template image")
#ap.add_argument("-i", "--images", required=True,
#	help="Path to images where going to be save")
#ap.add_argument("-v", "--visualize",
#	help="Flag indicating whether or not to visualize each iteration")
#args = vars(ap.parse_args())
#img = args["template"]

img_path = "data/test3.jpg"
img_rows, img_cols = 24, 24


model = load_model('model.h5')
model.load_weights('weights.h5')

# load the image image, convert it to grayscale
template = cv2.imread(img_path)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

delta = [1, 0.75, 0.5, 0.25, 0.1] #np.linspace(1, 0.1, 10, endpoint=True)
div = int(template.shape[0]/100) + 1

import time
start = time.time()

    
crop_image = [preprocessing.scale(cv2.resize(template, (24, 24)))]
crop_boxes = [[0, 0, 1.0]]
for scale in delta:
    # Resize the image according to the scale, and keep track of the ratio of the resizing 
    resized = imutils.resize(template, width = int(template.shape[1] * scale))
        
    if resized.shape[0] < 24 or resized.shape[1] < 24:
        break
    
    h = resized.shape[0]#np.linspace(0,resized.shape[0]-24, div).astype(int)
    w = resized.shape[1]#np.linspace(0,resized.shape[1]-24, div).astype(int)
    for y in range(0,h,div):
        for x in range(0,w,div):
            crop_img = resized[y:y+24, x:x+24]
            if crop_img.shape[0] < 24 or crop_img.shape[1] < 24:
                crop_img = cv2.resize(crop_img, (24, 24))
            crop_image.append(preprocessing.scale(crop_img))
            crop_boxes.append([x,y,scale])
    

end = time.time()
print("Tiempo de division de imagenes", end - start)

crop_image = np.array(crop_image)
crop_image = crop_image.reshape(crop_image.shape[0],img_rows, img_cols, 1)

start = time.time()
prediction = model.predict(crop_image)
end = time.time()
print("Tiempo de predicción", end - start)


start = time.time()
predict_faces=[]
for i, a in enumerate(prediction):
    if prediction[i] > 0.99:
        predict_faces.append([crop_image[i],crop_boxes[i], prediction[i]])

template = cv2.imread(img_path)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

img = template
faces_or_scale =[]
for a in predict_faces:
    x = int(a[1][0]/a[1][2])
    y = int(a[1][1]//a[1][2])
    x1 = int(x + 24/a[1][2])
    y1 = int(y + 24/a[1][2])
    faces_or_scale.append([x,y,x1,y1, x1+(x1-x), y1+(y1-y)])
    img = cv2.rectangle(img, (x, y), (x1,y1), (255,255,255), 5)

%varexp --imshow img

template = cv2.imread(img_path)
img = template
faces_or_scale_np = np.array(faces_or_scale)
paint_faces = []
for a in faces_or_scale:
    c = np.logical_and(np.abs(faces_or_scale_np[:,-2] - a[-2]) < int(template.shape[0]/10),
                       np.abs(faces_or_scale_np[:,-1] - a[-1]) < int(template.shape[0]/10))
    paint_faces.append(np.average(faces_or_scale_np[c], axis=0))

from random import randint
for a in paint_faces:
    img = cv2.rectangle(img, (int(a[0]), int(a[1])), (int(a[2]),int(a[3])), (255,255,255), 5)

end = time.time()
print("Tiempo de dibujado de boxes", end - start)

%varexp --imshow img


