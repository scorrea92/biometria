#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:09:30 2018

@author: sebastiancorrea
"""

import glob
import dlib
import cv2
import openface
import numpy as np
from sklearn import preprocessing

predictor_model = "shape_predictor_68_face_landmarks.dat"
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)


"""Faces Data Base"""
num = 0
participants = sorted(glob.glob("../datasets/img_align_celeba/*"))
faces = []
for participant in participants:
    image = cv2.imread(participant)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), participant))
    
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges

        face = image[np.abs(face_rect.top()):np.abs(face_rect.bottom()),
                            np.abs(face_rect.left()):np.abs(face_rect.right())]
        face = cv2.resize(face, (24, 24))
        face = (face-face.mean())/face.std()
        noalign = "../datasets/img_align_celeba_crop_face/"+str(num)+".jpg"

        # # Save the aligned image to a file
        #cv2.imwrite(noalign, face)
        faces.append(face)

        num += 1

faces = np.array(faces)
np.save("extra_faces.npy", faces)

"""Not Faces Data Base"""
num = 0
participants = sorted(glob.glob("../datasets/spatial_envelope_256x256_static_8outdoorcategories/*"))
not_faces = []
for participant in participants:
    image = cv2.imread(participant)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)
    h = np.linspace(0,image.shape[0]-24, 10).astype(int)
    w = np.linspace(0,image.shape[1]-24, 10).astype(int)
    for y in h:
        for x in w:
            crop_img = image[y:y+24, x:x+24]
            img = cv2.resize(crop_img, (24, 24))
            not_faces.append(preprocessing.scale(img))
            num += 1
            print("Found not-faces in the image file {}".format(participant))   

not_faces = np.array(not_faces)
np.save("extra_Notfaces.npy", not_faces)




