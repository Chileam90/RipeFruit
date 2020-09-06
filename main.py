import cv2
from utilities.Sliders import Sliders

sliders = Sliders("sliders", 5)

cv2.waitKey(0)

print(sliders.get_value_by_index(0))

cv2.destroyAllWindows()