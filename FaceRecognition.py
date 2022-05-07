import cv2 as cv
import numpy as np
from numba import jit
from matplotlib import pyplot as plt

img = cv.imread("testimage.png")
cv.imshow("Test",img)
cv.waitKey(0)