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

#Find Contours 
ret, thresh = cv2.threshold(tight, 40, 40, 20 )

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, -9, (2555,255,255), 2)
#cv2.imshow('tresh',th3)

#Min max cordinaten van canny edge
pts = np.argwhere(tight>0)
y1,x1 = pts.min(axis=0)
y2,x2 = pts.max(axis=0) 

midx = (x2 + x1) / 2
midy = (y2 + y1) / 2

x11 = int(midx - 40)
x22 = int(midx + 40)
y11 = int(midy - 40)
y22 = int(midy + 40)

print(pts.max(axis=0))
print(pts.min(axis=0))

# Knip de regio uit
cropped = image[y11:y22, x11:x22] 
cv2.imshow('edgs',cropped)

cv2.imwrite("croppedres.png", cropped)
average = cropped.mean(axis=0).mean(axis=0)
tagged = cv2.rectangle(image.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)

#saturatie
croppedsat = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV).astype("float32")
(h, s, v) = cv2.split(croppedsat)
s = s*20
s = np.clip(s,0,255)
croppedsat = cv2.merge([h,s,v])
croppedfin = cv2.cvtColor(croppedsat.astype("uint8"), cv2.COLOR_HSV2BGR)
cv2.imshow('edgs',croppedfin)

# vindt gemiddelde kleur
avg_color_per_row = np.average(croppedfin, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print(avg_color)

if avg_color[2] > 100 and avg_color[0] < 10 and avg_color[2] < 120:
    print('dit is een perzik')
if avg_color[2] > 120 and avg_color[0] < 10:
    print('dit is een banaan')
# vindt dominante kleur
pixels = np.float32(croppedfin.reshape(-1, 3))
n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)
dominant = palette[np.argmax(counts)]
print(dominant)



#Post image 
cv2.imshow('edges',tagged)

cv2.waitKey()
cv2.destroyAllWindows()