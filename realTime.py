
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import random

import time
from tqdm.notebook import tqdm
# from keras.preprocessing.image import load_img
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import cv2


# TRAIN_DIR = 'dataset1/train'
# TEST_DIR = 'dataset1/test'
IMAGE_SIZE = 48
class_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
label_to_class = {v: k for k, v in class_labels.items()}

label_colors = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 255, 255),  # Yellow
    'fear': (255, 0, 0),       # Blue
    'happy': (0, 255, 0),      # Green
    'neutral': (255, 255, 0),  # Cyan
    'sad': (255, 0, 255),      # Magenta
    'surprise': (0, 255, 255)  # Light Blue

}

# model = tf.keras.models.load_model('cnnmodel.h5')
model = keras.models.load_model('cnnmodel.h5')

# print in green, model imported
print('\033[92m' + 'Model Imported Successfully' + '\033[0m')



haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


cam = cv2.VideoCapture(0)
if(cam.isOpened()):
    print("video opened")
else:
    # print in red
    print('\033[91m' + 'Error in opening video' + '\033[0m')
time.sleep(2)
while True:
    i, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

        
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        face = np.array(face)       
        face = face/255.0
        face = face.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

        label = model.predict(face).argmax()
        label = class_labels[label]
        rectangle_color = label_colors.get(label, (0, 0, 0))
        cv2.rectangle(img, (x, y), (x+w, y+h), rectangle_color, 2)
        
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, rectangle_color, 2)
    cv2.imshow('img', img)
    cv2.imwrite('output.jpg', img)
    cv2.waitKey(27)