# -*- coding:utf-8 -*-

# Created by hrwhisper on 2016/4/24.


from skimage.filters import threshold_otsu
import skimage.io
from skimage import transform

sameSize = 16


def read_image(filename):
    image = skimage.io.imread(filename, as_grey=True)
    threshold = threshold_otsu(image)
    return image < threshold


def _find_min_range(image):
    up, down, left, right = image.shape[0], 0, image.shape[1], 0
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            if image[i][j]:
                up, down, left, right = min(up, i), max(down, i), min(left, j), max(right, j)
    return up, down, left, right


def to_the_same_size(image):
    # up, down, left, right = _find_min_range(image)
    # image = image[up:down, left:right]
    image = transform.resize(image, (sameSize, sameSize))
    return image
