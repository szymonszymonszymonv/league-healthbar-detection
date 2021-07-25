import numpy as np
import cv2 as cv


class health_bar_detector:
    avatar_shape = 75  # avatar is a square 75x75
    avatar_coords = [(2, 153), (2, 257), (2, 359), (2, 462), (2, 566), (1843, 153), (1843, 257), (1843, 359), (1843, 462), (1843, 566)]

    # upper and lower ranges of red and blue colors of health bars
    upper_blue = np.array([102, 243, 251])
    lower_blue = np.array([98, 134, 195])
    upper_red = np.array([5, 255, 255])
    lower_red = np.array([2, 123, 132])

    # required to make morphological transformations
    kernel = np.ones((2, 2), np.uint8)

    def __init__(self):
        self.image_name = input("ENTER IMAGE NAME: ")
        self.image = cv.imread(self.image_name)

    # remove avatars (they contain health bars, which may cause trouble for the algorithm)
    def remove_avatars(self):
        for x, y in self.avatar_coords:
            cv.rectangle(self.image, (x, y), (x + self.avatar_shape, y + self.avatar_shape), (0, 255, 0), thickness=-1)

    # only red and blue colors are required to find a health bar
    def create_mask(self):
        self.remove_avatars()
        hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        red_mask = cv.inRange(hsv, self.lower_red, self.upper_red)
        blue_mask = cv.inRange(hsv, self.lower_blue, self.upper_blue)
        mask = red_mask + blue_mask
        result = cv.bitwise_and(self.image, self.image, mask=mask)
        return result  # an image with only red and blue colors in it

    # get rid of useless objects and make health bars more rectangle-like
    def apply_transformations(self):
        result = self.create_mask()
        gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray_result, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        thresh = cv.bitwise_not(thresh)  # FOREGROUND NEEDS TO BE WHITE

        # blank image will be used to store health bars
        self.blank_image = thresh * 0

        transformed = cv.dilate(thresh, self.kernel, iterations=1)
        transformed = cv.morphologyEx(transformed, cv.MORPH_OPEN, self.kernel, iterations=2)
        transformed = cv.morphologyEx(transformed, cv.MORPH_CLOSE, self.kernel, iterations=1)
        return transformed

    def find_health_bars(self):
        transformed = self.apply_transformations()
        contours, hierarchy = cv.findContours(transformed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        health_bars_count = 0
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            # choose only rectangles, ignore small, non-rectangle objects
            if (len(contour) <= 4 or len(contour) <= 6) and w * h >= 100:
                cv.rectangle(self.blank_image, (x, y), (x + w, y + h), 255, thickness=-1)

        # connect hp bars with mana bars to make one rectangle
        self.blank_image = cv.morphologyEx(self.blank_image, cv.MORPH_CLOSE, self.kernel, iterations=2)

        # find contours and return the number of health bars
        contours, hierarchy = cv.findContours(self.blank_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        health_bars_count = len(contours)
        return health_bars_count

    def display_message(self):
        print("HEALTH BARS DETECTED ON SCREEN: " + str(self.find_health_bars()))


# MODIFY THIS LINE TO SELECT A SCREENSHOT
query_image1 = cv.imread("screen4.png")

avatar_shape = 75  # avatar is a square 75x75
avatar_coords = [(2, 153), (2, 257), (2, 359), (2, 462), (2, 566), (1843, 153), (1843, 257), (1843, 359), (1843, 462), (1843, 566)]
for x, y in avatar_coords:
    cv.rectangle(query_image1, (x, y), (x + avatar_shape, y + avatar_shape), (0, 255, 0), thickness=-1)


hsv = cv.cvtColor(query_image1, cv.COLOR_BGR2HSV)
upper_blue = np.array([102, 243, 251])
lower_blue = np.array([98, 134, 195])
upper_red = np.array([5, 255, 255])
lower_red = np.array([2, 123, 132])
red_mask = cv.inRange(hsv, lower_red, upper_red)
blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
mask = red_mask + blue_mask
res = cv.bitwise_and(query_image1, query_image1, mask=mask)
res_blue = cv.bitwise_and(query_image1, query_image1, mask=blue_mask)
res_red = cv.bitwise_and(query_image1, query_image1, mask=red_mask)

res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(res_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)  # O TO TO
thresh = cv.bitwise_not(thresh)  # KEEP FOREGROUND IN WHITE

kernel = np.ones((2, 2), np.uint8)
result = cv.dilate(thresh, kernel)
result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel, iterations=2)
result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel, iterations=1)

blank_image = thresh * 0

contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
result_copy = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
health_bars = 0
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    if (len(contour) <= 4 or len(contour) <= 6) and w * h >= 100:
        # for point in contour:
        health_bars += 1
        cv.rectangle(result_copy, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        cv.rectangle(query_image1, (x, y), (x + w, y + h), (0, 0, 255), thickness=-1)
        cv.rectangle(blank_image, (x, y), (x + w, y + h), 255, thickness=-1)

blank_image = cv.morphologyEx(blank_image, cv.MORPH_CLOSE, kernel, iterations=2)
contours_2, hierarchy = cv.findContours(blank_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
blank_image = cv.cvtColor(blank_image, cv.COLOR_GRAY2BGR)
print("HEALTH BARS DETECTED ON SCREEN: " + str(len(contours_2)))

for contour in contours_2:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(blank_image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

# cv.imshow("thresh", thresh)
# cv.imshow("res_red", res_red)
# cv.imshow("res_blue", res_blue)
# cv.imshow('puste', blank_image)
cv.imshow("image", query_image1)
# cv.imshow("res", res)
# cv.imshow("result copy", result_copy)
cv.waitKey(0)