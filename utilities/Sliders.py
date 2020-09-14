import cv2


class Sliders:

    def __init__(self, window_name,  *args):
        self.sliders = []
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        for arg in args:
            if isinstance(arg, int):
                for i in range(arg):
                    slider_name = "slider_" + str(i + 1)
                    cv2.createTrackbar(slider_name, window_name, 0, 255, self._nothing)
                    self.sliders.append(slider_name)
            else:
                cv2.createTrackbar(arg, window_name, 0, 255, self._nothing)
                self.sliders.append(arg)

        cv2.resizeWindow(window_name, 600, len(self.sliders) * 10)

    def get_value_by_name(self, slider_name):
        return cv2.getTrackbarPos(slider_name, self.window_name)

    def get_value_by_index(self, index):
        slider_name = self.sliders[index]
        return cv2.getTrackbarPos(slider_name, self.window_name)

    def set_value_by_name(self, slider_name, value):
        cv2.setTrackbarPos(slider_name, self.window_name, value)

    def set_value_by_index(self, index, value):
        slider_name = self.sliders[index]
        cv2.setTrackbarPos(slider_name, self.window_name, value)

    def _nothing(self, event):
        pass


if __name__ == "__main__":
    # sliders = Sliders("sliders", 5)
    # named_sliders = Sliders("named_sliders", "jan", "piet", "klaas")
    # cv2.waitKey(0)
    #
    # print("\nValues from sliders are:")
    # for slider in sliders.sliders:
    #     print(slider + ": " + str(sliders.get_value_by_name(slider)))
    #
    # for i in range(len(sliders.sliders)):
    #     print(slider + ": " + str(sliders.get_value_by_index(i)))
    #
    # print("\nValues from named_sliders are:")
    # for slider in named_sliders.sliders:
    #     print(slider + ": " + str(named_sliders.get_value_by_name(slider)))

    HSV_sliders = Sliders("HSV sliders", "h", "s", "v")

    while True:
        h = HSV_sliders.get_value_by_name("h")
        s = HSV_sliders.get_value_by_name("s")
        v = HSV_sliders.get_value_by_name("v")

        print('h', h, 's', s, 'v', v)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
