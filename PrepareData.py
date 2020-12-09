import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
import random
######################################

DATADIR = 'D:\\tomato\\images\\training'
IMG_SIZE=100
CATEGORIES = ['Bacterial_spot','blight','healthy','Yellow_Leaf_Curl_Virus']
xtrain=[]
for category in CATEGORIES:
    path = os.path.join(DATADIR,category) #get into each directory of healtht and blight one at a time
    for img in os.listdir(path):  # os.listdir(path) returns a list of strings with names of files in path
        try:
            img_array = cv2.imread(os.path.join(path, img))  # convert to image matrix OR read image;                                                    # in this case, it is color reading
            img_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))# resize all the images into similar size
            xtrain.append([img_array, category]) #append image information and its label in one list
        except Exception as e:# if there is an error in reading or resizing the image
            print('error')
random.shuffle(xtrain)
X = []
y = []
for features,label in xtrain:
    X.append(features)
    if label==CATEGORIES[0]:
        lab=0
    elif label==CATEGORIES[1]:
        lab=1
    elif label==CATEGORIES[2]:
        lab=2
    elif label==CATEGORIES[3]:
        lab=3
    y.append(lab)
X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,3)

y= np.array(y)
########################################################save data
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
###################################open data
print('data is saved')