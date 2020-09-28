import cv2
import glob
import numpy as np

from utilities.Sliders import Sliders
from utilities import ExtendedCv2 as ext

filenames = glob.glob('raw_images/*/*.jpg')

i = 0
num_files = len(filenames) - 1
slider = Sliders("sliders", "block_size", "constant_c", "erosion_i", "dilation_i", "lower", "upper")
slider.set_value_by_name("block_size", 2)
slider.set_value_by_name("constant_c", 2)
slider.set_value_by_name("blur", 1)

while True:
    img = cv2.imread(filenames[i])
    width, height, _ = img.shape
    new_width = int(width * .2)
    new_height = int(height * .2)

    # original = cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_LINEAR)

    org_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    org_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    org_s = org_hsv[:,:,1]
    org_gray_blur = cv2.GaussianBlur(org_gray, (99, 99), 0)
    org_s_blur = cv2.GaussianBlur(org_s, (99, 99), 0)

    block_size = slider.get_value_by_index(0) if slider.get_value_by_index(0) % 2 else slider.get_value_by_index(0) + 1
    constant_c = slider.get_value_by_index(1) if slider.get_value_by_index(1) % 2 else slider.get_value_by_index(1) + 1

    adaptive_s_inv = cv2.adaptiveThreshold(org_s_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, constant_c)
    adaptive_inv = cv2.adaptiveThreshold(org_gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, constant_c)
    # adaptive_cnts, _ = cv2.findContours(adaptive_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # num_cnts = 0
    # for adaptive_cnt in adaptive_cnts:
    #     ### gelijk aan canny edge
    #     # arc_percentage = (slider.get_value_by_name("arc_percentage") / 10)
    #     # epsilon = arc_percentage * cv2.arcLength(adaptive_cnt, False)
    #     # approx_cnt = cv2.approxPolyDP(adaptive_cnt, epsilon, False)
    #     ##################################################################################
    #     if cv2.contourArea(adaptive_cnt) > (10000 * .4):
    #         print("area", cv2.contourArea(adaptive_cnt))
    #         cv2.drawContours(img, adaptive_cnt, -1, (255, 0, 0), 10)
    #         num_cnts += 1
    #
    # print("number of contours", num_cnts)

    lower = slider.get_value_by_name("lower")
    upper = slider.get_value_by_name("upper")
    mask = cv2.inRange(org_s_blur, lower, upper)

    dsize = (new_height, new_width)
    ext.resized_imshow(org_gray, dsize, "org_gray")
    ext.resized_imshow(org_s, dsize, "org_s")
    ext.resized_imshow(adaptive_s_inv, dsize, "adaptive_s_inv")
    ext.resized_imshow(adaptive_inv, dsize, "adaptive_inv")
    ext.resized_imshow(mask, dsize, "inrange mask")
    ext.resized_imshow(img, dsize, "img")

    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break
    elif key == ord('z'):
        i -= 1
    elif key == ord('x'):
        i += 1

    if i > num_files:
        i = 0
    elif i < 0:
        i = num_files
