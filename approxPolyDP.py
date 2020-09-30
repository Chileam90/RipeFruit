import cv2
import glob
import numpy as np

from utilities.Sliders import Sliders
from utilities import ExtendedCv2 as ext

filenames = glob.glob('raw_images/pear/*.jpg')
num_files = len(filenames) - 1
file_index = 0

slider = Sliders("sliders", "block_size", "constant_c", "s_channel_threshold", "canny_sigma", "contour_threshold")
hue_slider = Sliders("hue_slider", "hue_lower", "hue_upper", "s_lower", "s_upper")
slider.set_value_by_name("block_size", 150)
slider.set_value_by_name("constant_c", 20)
slider.set_value_by_name("s_channel_threshold", 40)
slider.set_value_by_name("contour_threshold", 0)

hue_slider.set_value_by_name("hue_lower", 1)
hue_slider.set_value_by_name("hue_upper", 255)
hue_slider.set_value_by_name("s_upper", 60)

avg_color = np.zeros(1, dtype=np.uint8)
dominant_color = np.zeros(1, dtype=np.uint8)

fruit_list = ["peach", "orange", "banana", "pear"]
end_result = ""
end_percentage = 0

old_approx = np.nan

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

    ext.resized_imshow(thresh, (new_height, new_width), "thresh")

    mask_img = cv2.GaussianBlur(img, (49, 49), 0)
    mask_img[:, :, 0] = cv2.bitwise_and(mask_img[:, :, 0], thresh)
    mask_img[:, :, 1] = cv2.bitwise_and(mask_img[:, :, 1], thresh)
    mask_img[:, :, 2] = cv2.bitwise_and(mask_img[:, :, 2], thresh)

    # show results
    ext.resized_imshow(mask_combined, (new_height, new_width), "mask_combined")
    ext.resized_imshow(mask_img, (new_height, new_width), "mask_img")

    # get H channel from HSV colors to create contours for each Hue range
    masked_img_hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV_FULL)
    masked_img_h = masked_img_hsv[:, :, 0]

    # get the upper and lower for each Hue range
    hue_lower = hue_slider.get_value_by_name("hue_lower")
    hue_upper = hue_slider.get_value_by_name("hue_upper")

    # create hue mask for each Hue range
    hue_mask = cv2.inRange(masked_img_h, hue_lower, hue_upper)
    hue_contours, _ = cv2.findContours(hue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ext.resized_imshow(hue_mask, (new_height, new_width), "hue_mask")

    for hue_cnt in hue_contours:
        rect = cv2.minAreaRect(hue_cnt)
        x, y, w, h = cv2.boundingRect(hue_cnt)

        if w > 500 and h > 500:
            # get boxed points of rectangle in contour
            x11 = x
            x22 = int(x + w)
            y11 = y
            y22 = int(y + h)
            cv2.rectangle(draw_img, (x11, y11), (x22, y22), (255, 0, 0), 5)

            thresh_tag = thresh[y11:y22, x11:x22]
            contours, _ = cv2.findContours(thresh_tag, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)

                ###
                # approxPolyDP
                ###
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                draw_img_tag = draw_img[y11:y22, x11:x22]
                print("is convex", cv2.isContourConvex(approx))
                cv2.drawContours(draw_img_tag, [approx], 0, (0), 5)

                if np.all(old_approx != approx):
                    old_approx = approx
                    print("number approx", len(approx))

            ext.resized_imshow(draw_img, (new_height, new_width), "draw_img")

    # scroll through images
    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break
    elif key == ord('z'):
        file_index -= 1
        old_approx = np.nan
        old_results = np.nan
    elif key == ord('x'):
        file_index += 1
        old_approx = np.nan
        old_results = np.nan

    if file_index > num_files:
        file_index = 0
    elif file_index < 0:
        file_index = num_files

    if key != -1:
        print("file name", filenames[file_index])
