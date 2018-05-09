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
    if width > W:
        imgScale = W/width
        newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
        newimg = cv2.resize(oriimg,(int(newX),int(newY)))
    else:
        
        imgScale = int(width/24)*24/width
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
center_faces=[]
for i, a in enumerate(prediction):
    if prediction[i] > 0.9:
        predict_faces.append([crop_image[i],crop_boxes[i], prediction[i]])
        x = int(crop_boxes[i][0]/crop_boxes[i][2])
        y = int(crop_boxes[i][1]/crop_boxes[i][2])
        center_faces.append([x+24, y+24])

center_faces = np.array(center_faces)

if center_faces.shape[0] > 0:
    center_faces = center_faces[center_faces[:, 0].argsort()]
    
    new_boxes = []
    for box in center_faces:
        x = box[0]
        y = box[1]
        count = 1
        for all_box in center_faces:
            if abs(box[0]-all_box[0]) < swing/10 and abs(box[1]-all_box[1]) < swing/10:
                x += all_box[0]
                y += all_box[1]
                count+=1
        
        new_boxes.append([int(x/count), int(y/count)])
    new_boxes = np.unique(np.array(new_boxes), axis=0)
    
    new_boxes2 = []
    for box in new_boxes:
        x = box[0]
        y = box[1]
        count = 1
        for all_box in new_boxes:
            if abs(box[0]-all_box[0]) < swing/10 and abs(box[1]-all_box[1]) < swing/10:
                x += all_box[0]
                y += all_box[1]
                count+=1
        
        new_boxes2.append([int(x/count), int(y/count)])
    new_boxes2 = np.unique(new_boxes2, axis=0)
    
    template_color = cv2.imread(img_path)
    template_color_resize = scale_width(template_color, 24*25)
    img = template_color_resize
    
    for a in new_boxes2:
        img = cv2.rectangle(img,(a[0]-24, a[1]+24), (a[0]+24,a[1]-24), (255,255,255), 5)
    
    end = time.time()
    print("Tiempo de dibujado de boxes", end - start)
    
    %varexp --imshow img
    
    
    #cv2.imshow('Face Detection',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #not_faces = []
    #for not_face in predict_faces:
    #    not_faces.append(not_face[0].reshape(24,24))
    #
    #not_faces = np.array(not_faces)
    
else:
    print("Not Faces Detected")
    
    
