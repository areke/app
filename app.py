import numpy as np
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.neighbors import NearestNeighbors
from vgg16 import VGG16
from keras.models import Model
import json
import cv2
app = Flask(__name__)
model = VGG16(weights='imagenet', include_top=False)
features = np.array(json.load(open('features.json', 'r'))).astype(np.float32)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        im = cv2.resize(np.array(json.loads(request.form['image'])).astype(np.float32), (102, 102))
        X = np.array([im])
        f = model.predict(X)
        print f.shape
        print f
        f = f.reshape(1, f.shape[1] * f.shape[2] * f.shape[3])
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(features)
        distances, indices = nbrs.kneighbors(f)
        return json.dumps(indices.tolist())



@app.route('/')
def hello_world():
    return 'Hello from Flask!'
if __name__ == '__main__':
    app.run()