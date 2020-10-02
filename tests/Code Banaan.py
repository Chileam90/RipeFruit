# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:55:30 2020

@author: quint
"""
import cv2
import numpy as np
import glob
from utilities.Sliders import Sliders

imagebanaan = cv2.imread('..\\raw_images\\banana\\01_03_banana.jpg')

#Resize image
scale_percent = 40
width = int(imagebanaan.shape[1] * scale_percent / 100)    
height = int(imagebanaan.shape[0] * scale_percent / 100)    
dim = (width, height)
image = cv2.resize(imagebanaan, dim, interpolation = cv2.INTER_AREA)


#sliders = Sliders("sliders", 1)
#Preprocessing
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(imgray, 9, 7, 7)
img = cv2.Canny(blur,5,8)

#Find Contours 
ret, thresh = cv2.threshold(img, 60, 40, cv2.THRESH_MASK )
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -9, (2555,255,255), 2)

#Post image 
cv2.imshow('edges',img)

cv2.waitKey()
print(sliders.get_value_by_index(0))
cv2.destroyAllWindows()