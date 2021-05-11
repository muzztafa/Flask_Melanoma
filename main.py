from __future__ import division, print_function
import os
import sys
import glob
import re
import numpy as np

import base64
from io import BytesIO
from PIL import Image
from io import StringIO
import cv2



# Flask utils
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Imports for Deep Learning
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import  Conv2D, Dense, Dropout, Flatten, Input, Add, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from PIL import Image
from skimage import transform
from numpy.random import seed

#INCEPTION V3
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
from tensorflow.keras.applications.inception_v3 import InceptionV3


#Define a flask app
app = flask.Flask("__main__")

model = load_model('InceptionV3_version2_3000sample.hdf5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# print("model summary: ",model.summary())
#model._make_predict_function()

def model_predict(img_path, model):
    np_image = Image.open(img_path)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3)) 
    np_image = np.expand_dims(np_image, axis=0)
    # np_image

    preds = model.predict(np_image)
    return preds



    # img = image.load_img(img_path, target_size=(224, 224))

    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
  
    # x = np.expand_dims(x, axis=0)
    


    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    

    # x = preprocess_input(x, mode='tf')
    

    # #print('X here: ',x)

    # preds = model.predict(x)
    # return preds

@app.route('/', methods=['GET'])
def index():
    return flask.render_template("index.html")

@app.route('/predict', methods=["GET","POST"])
def upload():
    if request.method == "POST":
        #Get the file from the post request
        fa = request.get_json()
        f=fa.get('imagedata') 
        # Save the file to /uploads
        f=f.replace('data:image/png;base64,','')
        f=f.replace('data:image/jpeg;base64,','')
        f=f.replace('data:image/jpg;base64,','')
        #print(f)
        im_bytes = base64.b64decode(f)   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        np_image = Image.open(im_file)   # img is now PIL Image object
        print("hello")
        np_image = np.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (224, 224, 3))
        np_image = np.expand_dims(np_image, axis=0)
        # np_image

        preds = model.predict(np_image)

        print("preds::  ",preds)
       

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = tf.keras.applications.inception_v3.decode_predictions(
         #    preds, top=5
#)[0]                                                       # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        result = str(preds[0][0])               # Convert to string
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)  
