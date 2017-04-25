# -*-coding:utf-8 -*-

# For Capturing warnings on terminal
import logging        
logging.captureWarnings(True)

import cv2

import matplotlib.pyplot as plt
from skimage import  measure
from math import floor, ceil
from OneCharacterRecognize import OneCharacterRecognize
from correctWord import CorrectWord
from handwriting_tools import read_image, to_the_same_size

wordSpace = 7 #originally wordSpace = 7


def get_image_words(image):
    # Delete the included area，Return to the correct area
    def remove_range(cells):
        # b in a
        def range_include(a, b):
            return b.up >= a.up and b.down <= a.down and b.left >= a.left and b.right <= a.right

        def range_cmp(range_data_a, range_data_b):
            return -1 if range_data_a.down - range_data_a.up < range_data_b.down - range_data_b.up else 1

        cells.sort(range_cmp)
        n = len(cells)
        ok = [True] * n
        for i in xrange(1, n):
            for j in xrange(i):
                if ok[j] and range_include(cells[i], cells[j]):
                    ok[j] = False
        new_cells = [cells[i] for i in xrange(n) if ok[i]]
        return new_cells

        # Word sorting

    def mycmp(range_data_a, range_data_b):
        return -1 if range_data_a.left < range_data_b.left else 1

    contours = measure.find_contours(image, 0.8)
    cells = []
    for contour in contours:
        up, down, left, right = min(contour[:, 0]), max(contour[:, 0]), min(contour[:, 1]), max(contour[:, 1])
        if down - up >= wordSpace or right - left >= wordSpace:
            cells.append(RangeData(up, down, left, right))

    cells = remove_range(cells)
    cells.sort(mycmp)
    return cells


class RangeData(object):
    def __init__(self, up, down, left, right):
        self.up = floor(up)
        self.down = ceil(down)
        self.left = floor(left)
        self.right = ceil(right)

    def __str__(self):
        return ' '.join(str(i) for i in [self.left])

    __repr__ = __str__

    def get_info(self):
        return self.up, self.down, self.left, self.right


# if __name__ == '__main__':
#     def main():
#         plt.gray()
#         clf = OneCharacterRecognize()
#         print 'read test file'
#         image = read_image('./data/crop14.jpg')
#         word = get_image_words(image)
#         print len(word)
#         to_predict = []
#         for i, c in enumerate(word):
#             nw, sw, ne, se = c.get_info()
#             cur = to_the_same_size(image[nw:sw, ne:se])
#             ax = plt.subplot(len(word), 1, i + 1)
#             ax.imshow(cur)
#             ax.set_axis_off()
#             to_predict.append(cur.ravel())

#         ans = clf.predict(to_predict)
#         ans = ''.join(ans)
#         print ans
#         cw = CorrectWord()
#         print cw.correct(ans.lower())
#         plt.show()

#     main()

def my_recognizer(image_name):
    clf = OneCharacterRecognize()
    image = read_image('./' + image_name + '.jpg')
    word = get_image_words(image)
    print len(word)
    to_predict = []
    for i, c in enumerate(word):
        nw, sw, ne, se = c.get_info()
        cur = to_the_same_size(image[nw:sw, ne:se])
        to_predict.append(cur.ravel())

    ans = clf.predict(to_predict)
    if(ans == 0):
        return 0
    ans = ''.join(ans)
    print ans
    return ans
    # cw = CorrectWord()
    # print cw.correct(ans.lower())
    # return cw.correct(ans.lower())

