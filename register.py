# Gradivo:
# https://realpython.com/python-ai-neural-network/
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://medium.com/@prantiksen4/training-a-neural-network-with-histogram-of-oriented-gradients-hog-features-373b97be5971
# https://github.com/pransen/ComputerVisionAlgorithms/blob/master/Principal%20Component%20Analysis/PCA.ipynb

# https://www.astesj.com/publications/ASTESJ_060285.pdf

import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import paths
import os, sys
import pickle
from facerecognitionutility import facerecognition as fr

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

def objToFile(dir,obj):
    with open(dir, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fileToObj(dir):
    with open(dir, 'rb') as handle:
        return pickle.load(handle)

def updatePickles(userId, imagePaths, lbp_hogs, labels):            #Funkcija zgenerira nove značilnice, če so izpolnjeni pogoji
    for i in imagePaths:
        img = cv2.imread(i,0)                                           #Preberemo sivinsko sliko
        img = cv2.resize(img,(300,300),interpolation=cv2.INTER_AREA)    #Resize na 300x300
        gradients, directions = fr.sobel(img)
        imgLbp = fr.lbp(img).tolist()
        imgHog = fr.hog(img,8,12,2,gradients,directions)
        join = imgLbp+imgHog
        print("Processed image",i)
        lbp_hogs.append(join)
        labels.append(userId)
    print(len(lbp_hogs))
    print(len(labels))
    objToFile(modelsFolder+"/faces.pickle",lbp_hogs)
    objToFile(modelsFolder+"/labels.pickle",labels)
    return lbp_hogs, labels

userId = "nik"
conditions = True                                                #Pogoj za izvajanje bo v prihodnosti ALI je uporabnik že zabeležen v sistemu face-recognition prijave, če je, bo ta vrednost false

if(conditions):
    dirname = os.path.dirname(os.path.abspath(__file__))              #Pridobimo trenutni delovni direktorij
    imagesFolder = os.path.join(dirname,'Images')                     #Dobimo pot do Images mape, tukaj je lahko v prihodnosti več podmap za uporabnike?
    modelsFolder = os.path.join(dirname,'Models')
    imagePaths = list(paths.list_images(imagesFolder))                #Pridobimo poti do vseh slik v en array

    lbp_hogs = fileToObj(modelsFolder+"/faces.pickle")                #Preberemo trenutne značilnice iz datoteke faces.pickle
    labels = fileToObj(modelsFolder+"/labels.pickle")                 #Preberemo trenutne labele iz labels.pickle

    lbp_hogs, labels = updatePickles(userId, imagePaths, lbp_hogs, labels)    #Vnesemo novega uporabnika v zbirko značilnic

    #Ker se je v naši množici podatkov pojavila sprememba, moramo zato na novo zgenerirati model nevronskih mrež.

    #X_train, X_test, y_train, y_test = train_test_split(lbp_hogs, labels, test_size=0.20, random_state=42)      #Značilnice razdelimo na učno in testno množico

    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)             #Pripravimo Multi-Layer Perception Classifier
    mlp.fit(lbp_hogs,labels)
    objToFile(modelsFolder+"/model.pickle", mlp)                      #Shranimo zgeneriran model v datoteko model.pickle
else:
    print("This user is already registered in the system.")