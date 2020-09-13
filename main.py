import cv2
import glob
import numpy as np

from utilities.Sliders import Sliders

filenames = glob.glob('raw_images/banana/*.jpg')

i = 0
num_files = len(filenames) - 1
slider = Sliders("sliders", "threshold", "block_size", "constant_c", "canny_low", "arc_percentage", "blur")
slider.set_value_by_name("block_size", 2)
slider.set_value_by_name("constant_c", 2)
slider.set_value_by_name("blur", 1)

while True:
    img = cv2.imread(filenames[i])
    width, height, _ = img.shape
    new_width = int(width * .4)
    new_height = int(height * .4)

    original = cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_LINEAR)

    k = slider.get_value_by_index(5) if slider.get_value_by_index(5) % 2 else slider.get_value_by_index(5) + 1

    org_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    org_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV_FULL)
    org_gray = org_hsv[:,:,2]
    org_gray_blur = cv2.GaussianBlur(org_gray, (5, 5), 0)

    block_size = slider.get_value_by_index(1) if slider.get_value_by_index(1) % 2 else slider.get_value_by_index(1) + 1
    constant_c = slider.get_value_by_index(2) if slider.get_value_by_index(2) % 2 else slider.get_value_by_index(2) + 1

    adaptive = cv2.adaptiveThreshold(org_gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant_c)
    adaptive_inv = cv2.adaptiveThreshold(org_gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant_c)

    # org_gray_blur = cv2.GaussianBlur(adaptive_inv, (k, k), 0)
    # adaptive_inv = cv2.adaptiveThreshold(org_gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
    #                                     block_size, constant_c)

    # adaptive_cnts, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # adaptive_cnt = max(adaptive_cnts, key=cv2.contourArea)
    adaptive_cnts, _ = cv2.findContours(adaptive_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_cnts = 0

    for adaptive_cnt in adaptive_cnts:
        ### gelijk aan canny edge
        # arc_percentage = (slider.get_value_by_name("arc_percentage") / 10)
        # epsilon = arc_percentage * cv2.arcLength(adaptive_cnt, False)
        # approx_cnt = cv2.approxPolyDP(adaptive_cnt, epsilon, False)
        ##################################################################################
        if cv2.contourArea(adaptive_cnt) > 15000:
            print("area", cv2.contourArea(adaptive_cnt))
            cv2.drawContours(original, adaptive_cnt, -1, 255, 3)
            num_cnts += 1

    print("number of contours", num_cnts)
    cv2.imshow("org_gray_blur", org_gray_blur)
    cv2.imshow("original", original)
    cv2.imshow("org_gray", org_gray)
    cv2.imshow("adaptive", adaptive)
    cv2.imshow("adaptive_inv", adaptive_inv)

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
