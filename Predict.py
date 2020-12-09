from tkinter import*
from tkinter import ttk
from tkinter import filedialog
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

####################################import model
model=keras.models.load_model('mymodel.h5')

class Root(Tk):
    def __init__(self):
        super(Root,self).__init__()
        self.title('Title')
        self.minsize(640,400)
        self.labelFrame=ttk.LabelFrame(self,text='Enter image from files')
        self.labelFrame.grid(column=0,row=1,padx=20,pady=20)
        #self.button()
    def button(self):
        self.button=ttk.Button(self.labelFrame,text="Browse",command=self.fileDialog)
        self.button.grid(column=1,row=1)
        
    def fileDialog(self):
        self.filename=filedialog.askopenfilename(initialdir = "D:\\tomato\\images\\training",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.label=ttk.Label(self.labelFrame,text="")
        self.label.grid(column=1,row=2)
        ############################test code
        model=keras.models.load_model('mymodel.h5')
        CATEGORIES = ['Bacterial_spot',"blight", "healthy",'Yellow_Leaf']
        def prepare(filepath):
            IMG_SIZE = 100  # 50 in txt-based
            img_array = cv2.imread(filepath)  # read in the ige, convert to grayscale
            img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
            return img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.
        Myimg=prepare(self.filename)
        Myimg=tf.cast(Myimg,tf.float32)

        prediction = model.predict([Myimg])
        pre=prediction[0]
        ind=np.where(pre==1)[0][0]
        print(pre)
        print(CATEGORIES[ind])
        ##############################################################
        self.label.configure(text="This tomato leaf is " +CATEGORIES[ind])
if __name__=="__main__":
    root=Root()
    root.title='leaf detector app'
    root.button()
    root.filename='mark'
    root.mainloop()
