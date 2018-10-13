import numpy as np
from sklearn.manifold import TSNE
import imageio
from vgg16 import VGG16
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from sklearn.neighbors import NearestNeighbors
import cv2

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

X = []
for i in range(188):
    im = np.array(cv2.resize(cv2.imread('imgs/' + str(i) + '.jpg'), (102, 102))).astype(np.float32)
    X.append(preprocess_input(im))
X = np.array(X)
print X.shape
model = VGG16(weights='imagenet', include_top=False)

features = model.predict(X)
features = features.reshape(188, 4608)
print features.shape
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(features)
distances, indices = nbrs.kneighbors(features)

print indices[185]


