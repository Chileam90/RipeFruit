########################################################################################################################
# -date 28-09-2020
# -authors Quinten Elbers en Chileam Bohnen

import cv2
import glob
import numpy as np

from utilities.Sliders import Sliders
from utilities import ExtendedCv2 as ext

# filenames = glob.glob('raw_images/banana_orange_peach_pear/*.jpg')
filenames = glob.glob('raw_images/fruit_group/*.jpg')
# filenames = glob.glob('raw_images/fruit_group_bowl/*.jpg')
num_files = len(filenames) - 1
file_index = 0

slider = Sliders("sliders", "block_size", "constant_c", "s_channel_threshold", "canny_sigma", "contour_threshold")
hue_slider = Sliders("hue_slider", "hue_lower", "hue_upper", "s_lower", "s_upper")
slider.set_value_by_name("block_size", 150)
slider.set_value_by_name("constant_c", 20)
slider.set_value_by_name("s_channel_threshold", 60)
slider.set_value_by_name("contour_threshold", 0)

hue_slider.set_value_by_name("hue_upper", 255)
hue_slider.set_value_by_name("s_upper", 60)

avg_color = np.zeros(1, dtype=np.uint8)
dominant_color = np.zeros(1, dtype=np.uint8)

end_result = ""
end_percentage = 0
old_approx = np.nan
old_results = np.nan

fruit_list = ["peach", "orange", "banana", "pear"]
hue_lower_list = [1, 13, 23, 40]
hue_upper_list = [13, 23, 40, 70]

hue_index = 0
hue_index_len = len(fruit_list) - 1

knn_set = np.array([[253, 0], [5, 5], [5, 3], [4, 2], [0, 1], [15, 27], [11, 12], [17, 16], [14, 14], [16, 16], [27, 28], [27, 28], [28, 30], [25, 29], [33, 32], [52, 55], [52, 52], [55, 53], [50, 50], [38, 45], [37, 45]], dtype=np.float32)
knn_response = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3], [3]], dtype=np.float32)

# create knn
knn = cv2.ml.KNearest_create()
knn.train(knn_set, cv2.ml.ROW_SAMPLE, knn_response)
print("file name", filenames[file_index])
print("expected fruit:", fruit_list[hue_index])

while True:
    img = cv2.imread(filenames[file_index])
    draw_img = img.copy()
    mask_img = img.copy()
    org_img = img.copy()
    width, height, _ = img.shape
    new_width = int(width * .2)
    new_height = int(height * .2)
    ###
    # create mask for background deletion
    ###
    # get S channel in HSV colors and blur with kernel (99, 99)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    img_s = img_hsv[:, :, 1]
    img_s_blur = cv2.GaussianBlur(img_s, (99, 99), 0)

    # threshold filter on S channel after blur
    s_channel_threshold = slider.get_value_by_name("s_channel_threshold")
    ret, threshold_s = cv2.threshold(img_s_blur, s_channel_threshold, 255, cv2.THRESH_TOZERO)

    # adaptive Threshold on S channel in HSV colors
    block_size = slider.get_value_by_index(0) if slider.get_value_by_index(0) % 2 else slider.get_value_by_index(0) + 1
    constant_c = slider.get_value_by_index(1) if slider.get_value_by_index(1) % 2 else slider.get_value_by_index(1) + 1
    adaptive_s = cv2.adaptiveThreshold(img_s_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant_c)

    # inrange mask on S channel in HSV colors
    s_lower = hue_slider.get_value_by_name("s_lower")
    s_upper = hue_slider.get_value_by_name("s_upper")
    mask = cv2.inRange(img_s_blur, s_lower, s_upper)

    # combine the three masks
    mask_combined = cv2.bitwise_xor(adaptive_s, mask)
    mask_combined = cv2.bitwise_xor(mask_combined, threshold_s)
    contour_threshold = slider.get_value_by_name("contour_threshold")
    ret, thresh = cv2.threshold(mask_combined, 0, 255, cv2.THRESH_BINARY)

    # show thresh
    ext.resized_imshow(thresh, (new_height, new_width), "thresh")
    # ext.resized_imshow(threshold_s, (new_height, new_width), "threshold_s")
    # ext.resized_imshow(adaptive_s, (new_height, new_width), "adaptive_s")
    # ext.resized_imshow(mask, (new_height, new_width), "mask inrange")

    ###
    # remove background from copied original image
    ###
    # blur original image with kernel (13, 13)
    mask_img = cv2.GaussianBlur(img, (13, 13), 0)
    # combine final mask with image
    mask_img[:, :, 0] = cv2.bitwise_and(mask_img[:, :, 0], thresh)
    mask_img[:, :, 1] = cv2.bitwise_and(mask_img[:, :, 1], thresh)
    mask_img[:, :, 2] = cv2.bitwise_and(mask_img[:, :, 2], thresh)

    # show results
    # ext.resized_imshow(mask_combined, (new_height, new_width), "mask_combined")
    ext.resized_imshow(mask_img, (new_height, new_width), "mask_img")

    # get H channel from HSV colors to create contours for each Hue range
    masked_img_hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV_FULL)
    masked_img_h = masked_img_hsv[:, :, 0]

    # get the upper and lower for each Hue range
    hue_lower = hue_lower_list[hue_index]
    hue_upper = hue_upper_list[hue_index]

    # create hue mask for each Hue range
    hue_mask = cv2.inRange(masked_img_h, hue_lower, hue_upper)
    contours, _ = cv2.findContours(hue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ext.resized_imshow(hue_mask, (new_height, new_width), "hue_mask")

    if len(contours) > 0:
        # get largest contour based on cv2.contourArea()
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        center, (w, h), rot = rect
        if w > 300 and h > 300:
            ###
            # approxPolyDP
            ###
            epsilon = 0.025 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(draw_img, [approx], 0, (0), 5)

            if np.all(old_approx != approx):
                old_approx = approx
                print("number approx", len(approx))

            # get boxed points of rectangle in contour
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # draw rectangle shape
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)

            # find centroid with moments
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            x11 = int(cx - 150)
            x22 = int(cx + 150)
            y11 = int(cy - 150)
            y22 = int(cy + 150)
            # get pixels in tag
            tag = org_img[y11:y22, x11:x22]
            # draw tag
            cv2.rectangle(draw_img, (x11, y11), (x22, y22), (0, 255, 0), 3, cv2.LINE_AA)

            ###
            # get average color from tag and transfer to HSV colors for kNN
            ###
            # get average color from tag
            avg_color_per_row = np.average(tag, axis=0)
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
            h_avg_color = hsv_avg_color[:, :, 0].astype(np.float32)

            ###
            # get dominant color from tag and transfer to HSV colors for kNN
            ###
            # get dominant color
            pixels = np.float32(tag.reshape(-1, 3))
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
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
            h_dominant_color = hsv_dominant_color[:, :, 0].astype(np.float32)

            # print("fruit:", fruit_list[hue_index], "average:", h_avg_color[0], "dominant:", h_dominant_color[0])
            newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
            newcomer[0][0] = h_avg_color[0]
            newcomer[0][1] = h_dominant_color[0]
            # print("newcomer", newcomer)
            ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

            if np.any(old_results != results):
                old_results = results
                print("result:  {}\n".format(results))
                print("neighbours:  {}\n".format(neighbours))
                print("distance:  {}\n".format(dist))

                print("fruit is:", fruit_list[int(results[0])])

        ext.resized_imshow(draw_img, (new_height, new_width), "draw_img")

    # scroll through images
    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:
        break
    elif key == ord('z'):
        file_index -= 1
    elif key == ord('x'):
        file_index += 1
    elif key == ord('a'):
        hue_index -= 1
        old_approx = np.nan
        old_results = np.nan
    elif key == ord('s'):
        hue_index += 1
        old_approx = np.nan
        old_results = np.nan


    if file_index > num_files:
        file_index = 0
    elif file_index < 0:
        file_index = num_files

    if hue_index > hue_index_len:
        hue_index = 0
    elif hue_index < 0:
        hue_index = hue_index_len

    if key != -1:
        print("file name", filenames[file_index])
        print("expected fruit:", fruit_list[hue_index])

    dominant_color = np.zeros(1, dtype=np.uint8)
    avg_color = np.zeros(1, dtype=np.uint8)

cv2.destroyAllWindows()
