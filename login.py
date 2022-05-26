import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import paths
import os, sys
import pickle

def objToFile(dir,obj):
    with open(dir, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fileToObj(dir):
    with open(dir, 'rb') as handle:
        return pickle.load(handle)

userId = "0123"
dirname = os.path.dirname(os.path.abspath(__file__))
modelsFolder = os.path.join(dirname,'Models')

mlp = fileToObj(modelsFolder+"/model.pickle")