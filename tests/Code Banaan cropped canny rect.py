# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:55:30 2020

@author: quint
"""
import cv2
import numpy as np
import glob
from utilities.Sliders import Sliders

imagebanaan = cv2.imread('..\\raw_images\\pear\\01_00_pear.jpg')

#Resize image
scale_percent = 40
width = int(imagebanaan.shape[1] * scale_percent / 100)    
height = int(imagebanaan.shape[0] * scale_percent / 100)    
dim = (width, height)
image = cv2.resize(imagebanaan, dim, interpolation = cv2.INTER_AREA)
image_copy = image
image2 = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

def auto_canny(image, sigma = 0.35):
    # compute the mediam of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) *v))
    edged = cv2.Canny(image, lower, upper)

    # return edged image
    return edged

#sliders = Sliders("sliders", 1)
#Preprocessing
imgray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(imgray, 2, 2, 1)
#img = cv2.Canny(blur,5,8)
wide = auto_canny(blur, 500)
tight = auto_canny(blur, 500)
auto = auto_canny(blur)
#Find Contours 
ret, thresh = cv2.threshold(tight, 40, 40, 2 )
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Now crop

#Post image 
cv2.imshow('edges',out)

cv2.waitKey()
p#rint(sliders.get_value_by_index(0))
cv2.destroyAllWindows()