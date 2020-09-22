import cv2
import glob
import numpy as np

from utilities.Sliders import Sliders
from utilities import ExtendedCv2 as ext

from matplotlib import pyplot as plt


def auto_canny(image, lower_range=0, sigma=0.35):
    # compute the mediam of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(lower_range, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return edged image
    return edged


filenames = glob.glob('raw_images/banana_orange_peach_pear/*.jpg')
num_files = len(filenames) - 1
file_index = 0

slider = Sliders("sliders", "block_size", "constant_c", "s_channel_threshold", "canny_sigma", "contour_threshold")
hue_slider = Sliders("hue_slider", "hue_lower", "hue_upper")
slider.set_value_by_name("block_size", 255)
slider.set_value_by_name("constant_c", 5)
slider.set_value_by_name("s_channel_threshold", 40)
slider.set_value_by_name("contour_threshold", 0)

hue_slider.set_value_by_name("hue_upper", 255)

while True:
    img = cv2.imread(filenames[file_index])
    draw_img = img.copy()
    org_img = img.copy()
    width, height, _ = img.shape
    new_width = int(width * .2)
    new_height = int(height * .2)

    # adaptive Threshold on S channel in HSV colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    img_s = img_hsv[:, :, 1]
    # img_v = img_hsv[:, :, 2]
    img_s_blur = cv2.GaussianBlur(img_s, (99, 99), 0)
    # img_v_blur = cv2.GaussianBlur(img_v, (5, 5), 0)

    # threshold filter on S channel after blur
    s_channel_threshold = slider.get_value_by_name("s_channel_threshold")
    ret, threshold_s = cv2.threshold(img_s_blur, s_channel_threshold, 255, cv2.THRESH_TOZERO)

    block_size = slider.get_value_by_index(0) if slider.get_value_by_index(0) % 2 else slider.get_value_by_index(0) + 1
    constant_c = slider.get_value_by_index(1) if slider.get_value_by_index(1) % 2 else slider.get_value_by_index(1) + 1
    adaptive_s = cv2.adaptiveThreshold(img_s_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant_c)

    mask_combined = cv2.bitwise_xor(adaptive_s, threshold_s)
    contour_threshold = slider.get_value_by_name("contour_threshold")
    ret, thresh = cv2.threshold(mask_combined, 250, 255, cv2.THRESH_BINARY_INV)

    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # find the biggest countour (c) by the area
    # cnt = max(contours, key = cv2.contourArea)
    # print("max cnt:", cnt)
    #
    # cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)

    img[:, :, 0] = cv2.bitwise_and(img[:, :, 0], thresh)
    img[:, :, 1] = cv2.bitwise_and(img[:, :, 1], thresh)
    img[:, :, 2] = cv2.bitwise_and(img[:, :, 2], thresh)

    ext.resized_imshow(thresh, (new_height, new_width), "thresh")
    ext.resized_imshow(adaptive_s, (new_height, new_width), "adaptive_s")
    ext.resized_imshow(threshold_s, (new_height, new_width), "threshold_s")
    ext.resized_imshow(mask_combined, (new_height, new_width), "mask_combined")
    ext.resized_imshow(img_s_blur, (new_height, new_width), "img_s_blur")

    ext.resized_imshow(img, (new_height, new_width), "test")

    masked_img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    masked_img_h = masked_img_hsv[:, :, 0]

    masked_img_h_blur = cv2.GaussianBlur(masked_img_h, (5, 5), 0)

    # hue_histr = cv2.calcHist([masked_img_h], [0], None, [256], [1, 256])
    #plt.plot(hue_histr)
    #plt.show()

    hue_lower = hue_slider.get_value_by_name("hue_lower")
    hue_upper = hue_slider.get_value_by_name("hue_upper")

    print("hue_lower", hue_lower)
    print("hue_upper", hue_upper)

    #img_resize = cv2.resize(org_img, (int(height * .4), int(width * .4)))

    hue_mask = cv2.inRange(masked_img_h_blur, hue_lower, hue_upper)
    contours, _ = cv2.findContours(hue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("len(contours)", len(contours))
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        print("rect.shape", rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)

        # find
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print("Centroid = ", cx, ", ", cy)
        x11 = int(cx - 150)
        x22 = int(cx + 150)
        y11 = int(cy - 150)
        y22 = int(cy + 150)
        cut = org_img[y11:y22, x11:x22]
        tagged = cv2.rectangle(org_img, (x11, y11), (x22, y22), (0, 255, 0), 3, cv2.LINE_AA)
        ext.resized_imshow(org_img, (int(height * .2), int(width * .2)), "org_img")
        #cv2.imshow("img_resize", img_resize)

        # vindt gemiddelde kleur
        avg_color_per_row = np.average(cut, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        print("avg_color", avg_color)

        if avg_color[2] > 90 and avg_color[0] < 10 and avg_color[2] < 112:
            print('dit is een perzik')
        if avg_color[2] > 118 and avg_color[0] < 10:
            print('dit is een banaan')
        if avg_color[2] < 60 and avg_color[1] < 60:
            print('dit is een peer')

        # vindt dominante kleur
        pixels = np.float32(cut.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        print("dominant color", dominant)

#cv2.imshow('edgs', cut)

    ext.resized_imshow(draw_img, (new_height, new_width), "draw_img")
    ext.resized_imshow(hue_mask, (new_height, new_width), "hue_mask")

    # scroll through images
    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break
    elif key == ord('z'):
        file_index -= 1
    elif key == ord('x'):
        file_index += 1

    if file_index > num_files:
        file_index = 0
    elif file_index < 0:
        file_index = num_files

cv2.destroyAllWindows()
