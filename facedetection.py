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


def scale_width(oriimg, W):
    height, width, depth = oriimg.shape
    imgScale = W/width
    newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
    newimg = cv2.resize(oriimg,(int(newX),int(newY)))
    
    return newimg

img_path = "data/test5.jpg"
img_rows, img_cols = 24, 24


model = load_model('model_extraData.h5')
model.load_weights('weights_model_extraData.h5')

# load the image image, convert it to grayscale
template_color = cv2.imread(img_path)
template_color_resize = scale_width(template_color, 24*25)
template = cv2.cvtColor(template_color_resize, cv2.COLOR_BGR2GRAY)


delta = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
delta = np.linspace(1, 0.05, 20, endpoint=True)
if template.shape[0]>template.shape[1]:
    swing = template.shape[1]
else:
    swing = template.shape[0]
    
divh = int(swing/100) 
divw = int(swing/100) 

import time
start = time.time()

    
crop_image = [preprocessing.scale(cv2.resize(template, (24, 24)))]
crop_boxes = [[0, 0, 1.0]]
for scale in delta:
    # Resize the image according to the scale, and keep track of the ratio of the resizing 
    resized = imutils.resize(template, width = int(template.shape[1] * scale))
        
    if resized.shape[0] < 24 or resized.shape[1] < 24:
        break
    
    h = resized.shape[0]-24#np.linspace(0,resized.shape[0]-24, div).astype(int)
    w = resized.shape[1]-24#np.linspace(0,resized.shape[1]-24, div).astype(int)
    for y in range(0,h,divh):
        for x in range(0,w,divw):
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
print("Tiempo de prediccion", end - start)


start = time.time()
predict_faces=[]
for i, a in enumerate(prediction):
    if prediction[i] < 0.1:
        predict_faces.append([crop_image[i],crop_boxes[i], prediction[i]])

img = template_color_resize
faces_or_scale =[]
for a in predict_faces:
    x = int(a[1][0]/a[1][2])
    y = int(a[1][1]//a[1][2])
    x1 = int((a[1][0] + 24)/a[1][2])
    y1 = int((a[1][1] + 24)/a[1][2])
    faces_or_scale.append([x,y,x1,y1, x1+(x1-x), y1+(y1-y)])
    img = cv2.rectangle(img, (x, y), (x1,y1), (255,255,255), 5)

end = time.time()
print("Tiempo de dibujado de boxes", end - start)

%varexp --imshow img

#cv2.imshow('Face Detection',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#template = cv2.imread(img_path)
#img = template
#faces_or_scale_np = np.array(faces_or_scale)
#paint_faces = []
#for a in faces_or_scale:
#    c = np.logical_and(np.abs(faces_or_scale_np[:,-2] - a[-2]) < int(template.shape[0]/10),
#                       np.abs(faces_or_scale_np[:,-1] - a[-1]) < int(template.shape[0]/10))
#    paint_faces.append(np.average(faces_or_scale_np[c], axis=0))
#
#from random import randint
#for a in paint_faces:
#    img = cv2.rectangle(img, (int(a[0]), int(a[1])), (int(a[2]),int(a[3])), (255,255,255), 3)
#
#
#not_faces = []
#for not_face in predict_faces:
#    not_faces.append(not_face[0].reshape(24,24))
#
#not_faces = np.array(not_faces)



