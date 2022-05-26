import cv2
import numpy as np
#from numba import jit
from matplotlib import pyplot as plt
from imutils import paths
import os, sys

def lbp(lbpImage):
    rows, cols = lbpImage.shape
    lbpFinal = np.zeros(shape=(rows, cols))
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
            lbpFinal[i, j] = pValue
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
def HOG(image, N, B, M, gradients, directions):
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

