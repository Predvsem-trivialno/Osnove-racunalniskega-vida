import cv2 as cv
import numpy as np
#from numba import jit
from matplotlib import pyplot as plt

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

img = cv.imread("testimage.png")

lbpImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
lbpArray = lbp(lbpImage)
cv.imshow("Test",lbpArray)
cv.waitKey(0)