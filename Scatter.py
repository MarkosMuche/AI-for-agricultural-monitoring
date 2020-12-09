from tkinter import*
from tkinter import ttk
from tkinter import filedialog
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import random
####################################import model
model=keras.models.load_model('mymodel.h5')
dir='D:\\tomato\\images\\test'
x=[]
images=os.listdir(dir)

for imageString in images:
    path = os.path.join(dir, imageString)
    try:
        image=cv2.imread(path)
        image=cv2.resize(image,(100,100))
        image=image.reshape(-1, 100, 100, 3)
        x.append(image)
    except:
        print('error')

random.shuffle(x)
z0=[]
y0=[]
y1=[]
z1=[]
y2=[]
z2=[]
y3=[]
z3=[]
count=0
for xx in x:
    #print(model.predict(xx))
    pre=model.predict(xx)[0];
    if pre[0]==1.:
        z0.append(4-3*random.random())
        y0.append(3-3*random.random())
        count+=1
    elif pre[1]==1.:
        z1.append(7-3*random.random())
        y1.append(3-3*random.random())
        count+=1
    elif pre[2]==1.:
        z2.append(5-3*random.random())
        y2.append(7-3*random.random())
        count += 1
    elif pre[3]==1.:
        z3.append(8-3*random.random())
        y3.append(6-3*random.random())
        count += 1
plt.scatter(y0,z0,label='bacterial spot',color='r',marker='*')
plt.scatter(y1,z1,label='blight',color='g',marker='o')
plt.scatter(y2,z2,label='healthy',color='b',marker='*')
plt.scatter(y3,z3,label='Yellow_Leaf_Curl_Virus',color='y',marker='o')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.title('Farm Mapping')
plt.show()