import cv2
import numpy as np


def resized_imshow(img, dsize, name=""):
    n_img = cv2.resize(img, dsize, cv2.INTER_LINEAR)
    return cv2.imshow(name, n_img)


