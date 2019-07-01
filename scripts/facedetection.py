import numpy as np
import argparse
import cv2
from keras.models import load_model 
import scipy
import time
 
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image to Scan")
ap.add_argument("-p", "--pixel", help="Value of pixel iteration crop search")
args = vars(ap.parse_args())
path = str(args["image"])
pix = 0
if args['pixel']:
    pix = int(args["pixel"])

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

img_path = path # "data/test1.jpg"
img_rows, img_cols = 24, 24

try:

    model = load_model('models/model_extraData.h5')
    model.load_weights('models/weights_model_extraData.h5')

    # load the image image, convert it to grayscale
    template_color = cv2.imread(img_path)
    template_color_resize = scale_width(template_color, 24*20)
    template = cv2.cvtColor(template_color_resize, cv2.COLOR_BGR2GRAY)

    delta = np.linspace(1, 0.07, 15, endpoint=True)

    if pix > 0:
        swing = pix
        divh = swing 
        divw = swing
    else:
        if template.shape[0]>template.shape[1]:
            swing = template.shape[1]
        else:
            swing = template.shape[0]
        
        divh = int(swing/100) 
        divw = int(swing/100) 

    start = time.time()

    crop_boxes = []
    crop_image = []
    for scale in delta:
        # Resize the image according to the scale 
        resized = scipy.misc.imresize(template, scale)
        
        if resized.shape[0] < 24 or resized.shape[1] < 24:
            break
        h = resized.shape[0]-24
        w = resized.shape[1]-24
        for y in np.arange(0,h,divh):
            for x in np.arange(0,w,divw):
                crop_img = resized[y:y+24, x:x+24]
                if crop_img.std() > 40:
                    if crop_img.shape[0] < 24 or crop_img.shape[1] < 24:
                        crop_img = cv2.resize(crop_img, (24, 24))
                    crop_image.append( (crop_img-np.mean(crop_img, axis=0))/np.std(crop_img, axis=0) )
                    #crop_image.append(crop_img)
                    crop_boxes.append([x,y,scale])

    end = time.time()
    print("Tiempo de division de imagenes", end - start)

    start = time.time()
    crop_image = np.array(crop_image)
    crop_image = crop_image.reshape(crop_image.shape[0],img_rows, img_cols, 1)
    prediction = model.predict(crop_image)
    end = time.time()
    print("Tiempo de prediccion", end - start)


    start = time.time()
    center_faces=[]
    for i, a in enumerate(prediction):
        if prediction[i] > 0.9:
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
        
        # template_color = cv2.imread(img_path)
        # template_color_resize = scale_width(template_color, 24*25)
        img = template_color_resize
        
        for a in new_boxes2:
            img = cv2.rectangle(img,(a[0]-24, a[1]+24), (a[0]+24,a[1]-24), (255,255,255), 5)
        
        end = time.time()
        print("Tiempo de dibujado de boxes", end - start)
        
        # %varexp --imshow img

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
    else:
        print("Not Faces Detected")
except:
     print("Path not a image or not a valid path")       
        
