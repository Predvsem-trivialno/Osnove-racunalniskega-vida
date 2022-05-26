import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import paths
import os, sys
import pickle
import facerecognitionutility as fr

def objToFile(dir,obj):
    with open(dir, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fileToObj(dir):
    with open(dir, 'rb') as handle:
        return pickle.load(handle)

dirname = os.path.dirname(os.path.abspath(__file__))
modelsFolder = os.path.join(dirname,'Models')

mlp = fileToObj(modelsFolder+"/model.pickle")

img = cv2.imread("loginAttempt.jpg",0)
img = cv2.resize(img,(300,300),interpolation=cv2.INTER_AREA)

lbp_hog = []

gradients, directions = sobel(img)
imgLbp = lbp(img).tolist()
imgHog = hog(img,8,12,2,gradients,directions)
join = imgLbp+imgHog
lbp_hog.append(join)

prediction = mlp.predict_proba(lbp_hog)
print(mlp.classes_)
print(prediction)