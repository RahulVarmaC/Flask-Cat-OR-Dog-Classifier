from fileinput import filename
import numpy as np
from flask import Flask,request,render_template, redirect,url_for
import tensorflow as tf
import keras
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import os

app=Flask(__name__)


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "D:/Flask/Uploads"

cnn_model=tf.keras.models.load_model("C:/Users/crahu/Documents/cats_and_dogs_model/")


@app.route("/")

def home():
    return render_template('index.html')

@app.route('/predict',methods=['post'])

def predict():
    img=request.files['image_file']
    filename = secure_filename(img.filename)
    upload_image_path = os.path.join(UPLOAD_FOLDER + img.filename)
    img.save(upload_image_path)
    #img.save(secure_filename(img.filename))
    img2 = tf.io.read_file(upload_image_path)
    img1 = tf.io.read_file(upload_image_path)
    # img1=tf.reshape(img,(1,128,128,3))
    img1 = tf.image.decode_jpeg(img1, channels=3)
    img1 = tf.image.resize(img1, [128,128])
    # print(img.shape)
    img1=img1/255.0
    img1=tf.reshape(img1,(1,128,128,3))
    val=cnn_model.predict(img1)

    val=val.tolist()
    val=list(np.concatenate(val).flat)
    print(val.index(max(val)))
    val=val.index(max(val))

    if val<=0.5:
        pred="its a cat!"
    elif val>=0.5:
        pred="its a dog!"

    return render_template('index.html' , prediction_text=" The model says that {} !".format(pred))



if __name__=="__main__":
    app.run(debug=True)