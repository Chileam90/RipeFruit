# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:55:30 2020

@author: quint
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
#import glob
#from utilities.Sliders import Sliders

imagebanaan = cv2.imread('..\\raw_images\\banana\\01_00_banana.jpg')

#Resize image
scale_percent = 40
width = int(imagebanaan.shape[1] * scale_percent / 100)    
height = int(imagebanaan.shape[0] * scale_percent / 100)    
dim = (width, height)
image = cv2.resize(imagebanaan, dim, interpolation = cv2.INTER_AREA)
image_copy = image

def auto_canny(image, sigma = 0.35):
    # compute the mediam of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) *v))
    edged = cv2.Canny(image, lower, upper)

    # return edged image
    return edged

#Preprocessing
imgray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(imgray, 2, 2, 1)
#th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

wide = auto_canny(blur, 500)
tight = auto_canny(blur, 500)
auto = auto_canny(blur)
cv2.imshow('Canny',tight)

#Find Contours 
ret, thresh = cv2.threshold(tight, 40, 40, 20 )

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


cv2.drawContours(image, contours, -1, (0,0,255), 2)
cv2.imshow('contours',thresh)/


#find
M = cv2.moments(im2)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print ("Centroid = ", cx, ", ", cy)
x11 = int(cx - 40)
x22 = int(cx + 40)
y11 = int(cy - 40)
y22 = int(cy + 40)
cut = image[y11:y22, x11:x22] 
tagged = cv2.rectangle(image.copy(), (x11,y11), (x22,y22), (0,255,0), 3, cv2.LINE_AA)
cv2.imshow('edgs',cut)
    
#saturatie

# vindt gemiddelde kleur
avg_color_per_row = np.average(cropped1, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print(avg_color)

if avg_color[2] > 90 and avg_color[0] < 10 and avg_color[2] < 112:
    print('dit is een perzik')
if avg_color[2] > 118 and avg_color[0] < 10:
    print('dit is een banaan')
if avg_color[2] < 60 and avg_color[1] < 60:
    print('dit is een peer')
    
# vindt dominante kleur
pixels = np.float32(cropped1.reshape(-1, 3))
n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)
dominant = palette[np.argmax(counts)]
print(dominant)

#Post image 
cv2.imshow('Tagging',tagged)

cv2.waitKey()

cv2.destroyAllWindows()