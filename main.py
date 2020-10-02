import cv2
import glob
import numpy as np

from utilities.Sliders import Sliders
from utilities import ExtendedCv2 as ext

img_peach = glob.glob("raw_images\\peach\\01_00_peach.jpg")
img_orange = glob.glob("raw_images\\orange\\01_00_orange.jpg")
img_banana = glob.glob("raw_images\\banana\\01_03_banana.jpg")
img_pear = glob.glob("raw_images\\pear\\01_05_pear.jpg")

slider = Sliders("sliders", "block_size", "constant_c", "s_channel_threshold", "contour_threshold")
slider.set_value_by_name("block_size", 150)
slider.set_value_by_name("constant_c", 3)
slider.set_value_by_name("s_channel_threshold", 40)
slider.set_value_by_name("contour_threshold", 0)

hue_slider = Sliders("hue_slider", "hue_lower", "hue_upper", "s_lower", "s_upper")
hue_slider.set_value_by_name("hue_upper", 255)
hue_slider.set_value_by_name("s_upper", 40)


def combine_images():
    img_0 = cv2.imread(img_peach[0])
    img_1 = cv2.imread(img_orange[0])
    img_2 = cv2.imread(img_banana[0])
    img_3 = cv2.imread(img_pear[0])

    img_h_0 = cv2.hconcat([img_0, img_1])
    img_h_1 = cv2.hconcat([img_2, img_3])
    n_img = cv2.vconcat([img_h_0, img_h_1])
    return n_img

cv2.erode

img = combine_images()
width, height, _ = img.shape
new_width = int(width * .1)
new_height = int(height * .1)
n_size = (new_height, new_width)

while True:
    img = combine_images()
    mask_img = img.copy()

    width, height, _ = img.shape
    new_width = int(width * .1)
    new_height = int(height * .1)
    n_size = (new_height, new_width)

    # # adaptive Threshold on S channel in HSV colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    img_s = img_hsv[:, :, 1]

    img_s_blur = cv2.GaussianBlur(img_s, (99, 99), 0)
    #
    # # threshold filter on S channel after blur
    s_channel_threshold = slider.get_value_by_name("s_channel_threshold")
    ret, threshold_s = cv2.threshold(img_s_blur, s_channel_threshold, 255, cv2.THRESH_TOZERO)
    #
    block_size = slider.get_value_by_index(0) if slider.get_value_by_index(0) % 2 else slider.get_value_by_index(0) + 1
    constant_c = slider.get_value_by_index(1) if slider.get_value_by_index(1) % 2 else slider.get_value_by_index(1) + 1
    adaptive_s = cv2.adaptiveThreshold(img_s_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant_c)
    #
    s_lower = hue_slider.get_value_by_name("s_lower")
    s_upper = hue_slider.get_value_by_name("s_upper")
    inrange_mask = cv2.inRange(img_s_blur, s_lower, s_upper)
    #
    mask_adapt_xor_inrange = cv2.bitwise_xor(adaptive_s, inrange_mask)
    mask_combined = cv2.bitwise_xor(mask_adapt_xor_inrange, threshold_s)
    # contour_threshold = slider.get_value_by_name("contour_threshold")
    ret, thresh = cv2.threshold(mask_combined, 0, 255, cv2.THRESH_BINARY)
    #
    # mask_img = cv2.GaussianBlur(img, (49, 49), 0)
    #
    mask_img[:, :, 0] = cv2.bitwise_and(img[:, :, 0], thresh)
    mask_img[:, :, 1] = cv2.bitwise_and(img[:, :, 1], thresh)
    mask_img[:, :, 2] = cv2.bitwise_and(img[:, :, 2], thresh)

    # show results
    ext.resized_imshow(img_s, n_size, "img_s")
    ext.resized_imshow(img, n_size, "original image")
    # ext.resized_imshow(img_s_blur, n_size, "original image blur")
    # ext.resized_imshow(adaptive_s, n_size, "adaptive threshold")
    # ext.resized_imshow(threshold_s, n_size, "threshold binary")
    # ext.resized_imshow(inrange_mask, n_size, "mask inRange")
    ext.resized_imshow(mask_adapt_xor_inrange, (new_height, new_width), "adaptiveThreshold XOR inRange")
    ext.resized_imshow(mask_combined, (new_height, new_width), "adaptiveThreshold XOR inRange XOR threshold")
    ext.resized_imshow(thresh, n_size, "Threshold on final mask")
    ext.resized_imshow(mask_img, n_size, "Final product")

    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break
