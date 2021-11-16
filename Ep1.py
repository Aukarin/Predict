from flask import Flask ,render_template,request
from keras.backend import argmax
import tensorflow as tf
from tensorflow  import keras
import keras
import matplotlib.pyplot as plt

import numpy as np
import cv2
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

app = Flask(__name__)


@app.route('/',methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)
    model = load_model("pima4_model.h5")
    
   

    image = cv2.imread(image_path)
    image = cv2.resize(image,(200,200))
    image = img_to_array(image)
    image = image.reshape(1,200*200*3)
    image = image.astype('float32')
    image /= 255
    Ans =  model.predict(image) 
    class_names = ['No', 'Yes']
    result = class_names[np.argmax(Ans)]
    
    



    return render_template('index.html',prediction=result, img = image_path)
    
if __name__ == '__main__':
    app.run(port=3000,debug=True)
