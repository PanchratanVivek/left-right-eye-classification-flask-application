from flask import Flask, render_template, request
#from keras.models import load_model
#from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image

app = Flask(__name__)

import numpy as np
import pandas as pd
from PIL import Image
#import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime
import time
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore') 
model=load_model(r"final_left_right.h5", compile=False)

import glob 


def predict_label(img_path):
    
    
    img4 = cv2.imread(img_path)

    test_image=image.load_img(img_path,target_size=(224,224))
    

    test_image=image.img_to_array(test_image)
    test_image= cv2.addWeighted(test_image,4,cv2.GaussianBlur(test_image,(0,0),300/100),-4,128)
    test_image=np.expand_dims(test_image,axis=0)

    result=model.predict(test_image)

    y_pred = np.argmax(result)
    o=(result[0][0])
    if o<.5:
        
        cv2.imwrite(rf"static\{os.path.basename(img_path)}",img4)
        return ("Left Eye")
    else:
        cv2.imwrite(rf"static\{os.path.basename(img_path)}",img4)
        return ("Right Eye")




# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path ="static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)