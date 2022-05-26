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

def lbp(lbpImage):
    rows, cols = lbpImage.shape
    lbpFinal = np.zeros(pow(2, 8))
    lbpFinal = lbpFinal.astype('uint8')

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            pValue = 0
            lbpArray = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            center = lbpImage[i, j]
            if(lbpImage[i - 1, j - 1] >= center): lbpArray[0] = 1
            if(lbpImage[i - 1, j] >= center): lbpArray[1] = 2
            if(lbpImage[i - 1, j + 1] >= center): lbpArray[2] = 4
            if(lbpImage[i, j + 1] >= center): lbpArray[3] = 8
            if(lbpImage[i + 1, j + 1] >= center): lbpArray[4] = 16
            if(lbpImage[i + 1, j] >= center): lbpArray[5] = 32
            if(lbpImage[i + 1, j - 1] >= center): lbpArray[6] = 64
            if(lbpImage[i, j - 1] >= center): lbpArray[7] = 128
            for k in range(0, lbpArray.size):
                pValue = pValue + lbpArray[k]
            lbpFinal[pValue] += 1
    return lbpFinal

def sobel(src):
    src = cv2.GaussianBlur(src, (3, 3), 0)
    grad_x = cv2.Sobel(src, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    dirs = np.empty_like(grad_x)
            
    dirs = np.arctan2(grad_y,grad_x) * 180 / np.pi
    for i in range(len(dirs)):
        for j in range(len(dirs[i])):
            if(dirs[i][j]<0):
                dirs[i][j]+=180
            if(dirs[i][j]==180):
                dirs[i][j]=179
            dirs[i][j] = int(dirs[i][j])

    return grad.astype(np.uint8), dirs.astype(np.uint8)

def adjustSize(image, N):
    (height, width) = image.shape
    addWidth = (N-(width%N))         #Prilagodimo širino in višino, da lahko dobimo točno prileganje glede na celice.
    addHeight = (N-(height%N))       #Gledamo ostanek po deljenju z velikostjo celice - povečamo velikost okna do naslednje številke, ki je deljiva z našo velikostjo celice
    image = cv2.copyMakeBorder(image, 0, addHeight, 0, addWidth, cv2.BORDER_REPLICATE)  #Dodamo "border", razširimo sliko s podvojenimi robnimi piksli
    return image

def normalize(numbers):
    sum = 0
    for i in numbers:
        sum+=(i*i)
    k = np.sqrt(sum)    #Izračunamo konkatenacijski faktor
    new=[]
    for j in numbers:
        if (j==0 or k==0):
            new.append(0)
        else: 
            new.append(j/k)
    return new      #Vrnemo posodobljene vrednosti

#N je velikost celice, B je število košev, M je velikost bloka
def hog(image, N, B, M, gradients, directions):
    binGap = 180/B
    (height, width) = image.shape
    #0 15 30 45 60 75 90 105 120 135 150 165 - za B = 12
    bins = [[[0 for _ in range(int(B))] for _ in range(int(height/N))] for _ in range(int(width/N))]     #Pripravimo 3D array - 2D array seznamov, kamor shranjujemo vrednosti košev za posamezno celico
    #print("Število celic:",int(width/N)*int(height/N))
    for x in range(0, width-N, N):
        for y in range(0, height-N, N):
            cellG = gradients[y:y+N,x:x+N]              #Naredimo podcelico za gradiente
            cellA = directions[y:y+N,x:x+N]             #Naredimo podcelico za smeri gradientov
            bin = np.zeros(B,np.float64)                  #Pripravimo prazne koše                   #int
            for i in range(N):
                for j in range(N):
                    angle = cellA[j,i]                  #Preberemo kot iz matrike kotov gradientov
                    if(angle%binGap==0):
                        binIndex = int(angle/binGap)
                        bin[binIndex] += cellG[j,i]     #Če kot pade točno v nek koš
                    else:
                        binIndexLeft = int((angle-(angle%binGap))/binGap)           #Izračuna index levega in desnega koša na podlagi ostanka po deljenju
                        binIndexRight = int((angle-(angle%binGap)+binGap)/binGap)   #Ista enačba kot zgoraj, s tem da prištejemo še razmik med koši preden delimo
                        valueToLeft = 1 - (angle-(binIndexLeft*binGap))/binGap      #Izračunamo kolikšen deleš vrednosti bo padel v levi koš in kolikšen v desni
                        valueToRight = 1 - ((binIndexRight*binGap)-angle)/binGap
                        bin[binIndexLeft] += (cellG[j,i]*valueToLeft)    #V levi koš se vedno nekaj preslika, na desni strani pa lahko pridemo do prekoračitve pri zadnjem košu zato imam spodaj pogojni stavek      #int
                        if(angle>180-binGap):                               #Če je večje kot npr. 165
                            bin[0] += (cellG[j,i]*valueToRight)          #Se delež preslika v koš 0          #int
                        else:
                            bin[binIndexRight] += int(cellG[j,i]*valueToRight)  #Drugače se delež preslika v naslednjega        #int
            bins[int(x/N)][int(y/N)] = bin
                        
    output=[]
    for w in range(0, int(width/N)-M, 1):               #Pomikamo se skozi vse celice, pazimo da ne gremo čez rob (-M)
        for h in range(0, int(height/N)-M, 1):          #Po višini in širini
            concat=[]
            for subw in range(0, M, 1):                 #Na vsaki poziciji obdelamo M*M celic in združimo njihove bin tabele v eno
                for subh in range(0, M, 1):
                    concat.extend(bins[w+subw][h+subh])
            output.extend(normalize(concat))            #To združeno tabelo normaliziramo in jo pripišemo v output, kjer nastaja dolga datoteka
    return output

def updatePickles(userId, imagePaths, lbp_hogs, labels):            #Funkcija zgenerira nove značilnice, če so izpolnjeni pogoji
    for i in imagePaths:
        img = cv2.imread(i,0)                                           #Preberemo sivinsko sliko
        img = cv2.resize(img,(300,300),interpolation=cv2.INTER_AREA)    #Resize na 300x300
        gradients, directions = sobel(img)
        imgLbp = lbp(img).tolist()
        imgHog = hog(img,8,12,2,gradients,directions)
        join = imgLbp+imgHog
        print("Processed image",i)
        lbp_hogs.append(join)
        labels.append(userId)
    objToFile(modelsFolder+"/faces.pickle",lbp_hogs)
    objToFile(modelsFolder+"/labels.pickle",labels)
    return lbp_hogs, labels

userId = "0123"
conditions = False                                                #Pogoj za izvajanje bo v prihodnosti ALI je uporabnik že zabeležen v sistemu face-recognition prijave, če je, bo ta vrednost false

if(conditions):
    dirname = os.path.dirname(os.path.abspath(__file__))              #Pridobimo trenutni delovni direktorij
    imagesFolder = os.path.join(dirname,'Images')                     #Dobimo pot do Images mape, tukaj je lahko v prihodnosti več podmap za uporabnike?
    modelsFolder = os.path.join(dirname,'Models')
    imagePaths = list(paths.list_images(imagesFolder))                #Pridobimo poti do vseh slik v en array
    
    lbp_hogs = fileToObj(modelsFolder+"/faces.pickle")                #Preberemo trenutne značilnice iz datoteke faces.pickle
    labels = fileToObj(modelsFolder+"/labels.pickle")                 #Preberemo trenutne labele iz labels.pickle

    lbp_hogs, labels = updatePickles(imagePaths, lbp_hogs, labels)    #Vnesemo novega uporabnika v zbirko značilnic

    #Ker se je v naši množici podatkov pojavila sprememba, moramo zato na novo zgenerirati model nevronskih mrež.

    #X_train, X_test, y_train, y_test = train_test_split(lbp_hogs, labels, test_size=0.20, random_state=42)      #Značilnice razdelimo na učno in testno množico

    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)             #Pripravimo Multi-Layer Perception Classifier
    mlp.fit(lbp_hogs,labels)
    objToFile(modelsFolder+"/model.pickle", mlp)                      #Shranimo zgeneriran model v datoteko model.pickle
else:
    print("This user is already registered in the system.")