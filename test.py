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


filenames = glob.glob('raw_images/*/01_*.jpg')
num_files = len(filenames) - 1
file_index = 0

slider = Sliders("sliders", "block_size", "constant_c", "s_channel_threshold", "canny_sigma", "contour_threshold")
hue_slider = Sliders("hue_slider", "hue_lower", "hue_upper", "s_lower", "s_upper")
slider.set_value_by_name("block_size", 150)
slider.set_value_by_name("constant_c", 10)
slider.set_value_by_name("s_channel_threshold", 40)
slider.set_value_by_name("contour_threshold", 0)

hue_slider.set_value_by_name("hue_upper", 255)
hue_slider.set_value_by_name("s_upper", 60)

avg_color = np.zeros(1, dtype=np.uint8)
dominant_color = np.zeros(1, dtype=np.uint8)

fruit_list = ["peach", "orange", "banana", "pear"]
end_result = ""
end_percentage = 0

while True:
    img = cv2.imread(filenames[file_index])
    draw_img = img.copy()
    mask_img = img.copy()
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

    s_lower = hue_slider.get_value_by_name("s_lower")
    s_upper = hue_slider.get_value_by_name("s_upper")
    mask = cv2.inRange(img_s_blur, s_lower, s_upper)

    mask_combined = cv2.bitwise_xor(adaptive_s, mask)
    mask_combined = cv2.bitwise_xor(mask_combined, threshold_s)
    contour_threshold = slider.get_value_by_name("contour_threshold")
    ret, thresh = cv2.threshold(mask_combined, 0, 255, cv2.THRESH_BINARY)

    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # find the biggest countour (c) by the area
    # cnt = max(contours, key = cv2.contourArea)
    # print("max cnt:", cnt)
    #
    # cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)

    mask_img = cv2.GaussianBlur(img, (13, 13), 0)

    mask_img[:, :, 0] = cv2.bitwise_and(mask_img[:, :, 0], thresh)
    mask_img[:, :, 1] = cv2.bitwise_and(mask_img[:, :, 1], thresh)
    mask_img[:, :, 2] = cv2.bitwise_and(mask_img[:, :, 2], thresh)

    # ext.resized_imshow(mask, (new_height, new_width), "mask")
    # ext.resized_imshow(thresh, (new_height, new_width), "thresh")
    # ext.resized_imshow(adaptive_s, (new_height, new_width), "adaptive_s")
    # ext.resized_imshow(threshold_s, (new_height, new_width), "threshold_s")
    ext.resized_imshow(mask_combined, (new_height, new_width), "mask_combined")
    # ext.resized_imshow(img_s_blur, (new_height, new_width), "img_s_blur")

    ext.resized_imshow(mask_img, (new_height, new_width), "mask_img")

    masked_img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    masked_img_h = masked_img_hsv[:, :, 0]

    # masked_img_h_blur = cv2.GaussianBlur(masked_img_h, (13, 13), 0)

    # hue_histr = cv2.calcHist([masked_img_h], [0], None, [256], [1, 256])
    #plt.plot(hue_histr)
    #plt.show()

    hue_lower = hue_slider.get_value_by_name("hue_lower")
    hue_upper = hue_slider.get_value_by_name("hue_upper")

    # print("hue_lower", hue_lower)
    # print("hue_upper", hue_upper)

    #img_resize = cv2.resize(org_img, (int(height * .4), int(width * .4)))

    hue_mask = cv2.inRange(masked_img_h, hue_lower, hue_upper)
    contours, _ = cv2.findContours(hue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print("len(contours)", len(contours))
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        # print("rect.shape", rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)

        # find
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # print("Centroid = ", cx, ", ", cy)
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
        new_avg_color = np.average(avg_color_per_row, axis=0).astype(np.uint8)
        # opencv friendly array
        new_pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        # combine average with opencv friendly array
        new_pixel[:, :, 0] = new_avg_color[0]
        new_pixel[:, :, 1] = new_avg_color[1]
        new_pixel[:, :, 2] = new_avg_color[2]
        # transform to HSV color
        hsv_avg_color = cv2.cvtColor(new_pixel, cv2.COLOR_BGR2HSV_FULL)
        # get h channel
        h_avg_color = hsv_avg_color[:, :, 0]

        if np.all(h_avg_color[0] != avg_color):
            avg_color = h_avg_color[0]
            print("avg_color", avg_color)

            if avg_color[0] >= 1.0 and avg_color[0] < 13.0:
                end_result = fruit_list[0]
                end_percentage = 50
                print('dit is een perzik')
            elif avg_color[0] >= 13.0 and avg_color[0] < 22.0:
                end_result = fruit_list[1]
                end_percentage = 50
                print('dit is een sinaasapel')
            elif avg_color[0] >= 23.0 and avg_color[0] < 37.0:
                end_result = fruit_list[2]
                end_percentage = 50
                print('dit is een banaan')
            elif avg_color[0] >= 40.0 and avg_color[0] < 70.0:
                end_result = fruit_list[3]
                end_percentage = 50
                print('dit is een peer')
            else:
                end_result = ""
                continue

        # vindt dominante kleur
        pixels = np.float32(cut.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        new_dominant_color = palette[np.argmax(counts)]
        # opencv friendly array
        new_pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        new_pixel[:, :, 0] = new_dominant_color[0]
        new_pixel[:, :, 1] = new_dominant_color[1]
        new_pixel[:, :, 2] = new_dominant_color[2]
        # transform to HSV color
        hsv_dominant_color = cv2.cvtColor(new_pixel, cv2.COLOR_BGR2HSV_FULL)
        # get h channel
        h_dominant_color = hsv_dominant_color[:, :, 0]

        if np.all(dominant_color != h_dominant_color[0]):
            dominant_color = h_dominant_color[0]
            print("dominant color", dominant_color)
            if dominant_color[0] >= 1.0 and dominant_color[0] < 13.0:
                if end_result == fruit_list[0]:
                    end_percentage += 50
            elif dominant_color[0] >= 13.0 and dominant_color[0] < 22.0:
                if end_result == fruit_list[1]:
                    end_percentage += 50
            elif dominant_color[0] >= 23.0 and dominant_color[0] < 34.0:
                if end_result == fruit_list[2]:
                    end_percentage += 50
            elif dominant_color[0] >= 34.0 and dominant_color[0] < 50.0:
                if end_result == fruit_list[3]:
                    end_percentage += 50
            else:
                end_result = ""
                continue

        if end_percentage > 0:
            print('end result:', end_result, 'with ' + str(end_percentage) + ' % certainty')

        end_percentage = 0

#cv2.imshow('edgs', cut)

    ext.resized_imshow(draw_img, (new_height, new_width), "draw_img")
    ext.resized_imshow(hue_mask, (new_height, new_width), "hue_mask")

    # scroll through images
    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break
    elif key == ord('z'):
        file_index -= 1
        avg_color = np.zeros(1, dtype=np.uint8)
        dominant_color = np.zeros(1, dtype=np.uint8)
    elif key == ord('x'):
        file_index += 1
        avg_color = np.zeros(1, dtype=np.uint8)
        dominant_color = np.zeros(1, dtype=np.uint8)

    if file_index > num_files:
        file_index = 0
    elif file_index < 0:
        file_index = num_files

cv2.destroyAllWindows()
