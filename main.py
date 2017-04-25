import cv2
import imutils
from PIL import Image
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import pytesseract
from OneWordSolve import my_recognizer


def captch_ex(file_name ):

    img  = cv2.imread(file_name, 0)

    crop_image = cv2.imread(file_name, 0)
    crop_image = imutils.resize(crop_image, width=850)

    img = imutils.resize(img, width=850)

    white = np.zeros((478,850,1), np.uint8)
    white[:] = 255

    img_final = cv2.imread(file_name, 0)
 
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    img_final = imutils.resize(img_final, width=850)
    bin_image = cv2.adaptiveThreshold(img_final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,89,18)

    cv2.imwrite('bin.jpg', bin_image)

    result = 255-bin_image

    kernel = np.ones((2,2),np.uint8)
    result = cv2.dilate(result,kernel,iterations = 1)

    height, width = result.shape[:2]

    cv2.imshow('captcha_result' , result)
    cv2.waitKey()

    L = measure.label(result, background = 0)
    print "Number of components:", np.max(L)

    count = np.max(L)

    final = np.zeros((478,850,1), np.uint8)
    final[:] = 255

    for i in range(count):
      black = np.zeros((478,850,1), np.uint8)
      for x in range(height):
        for y in range(width):
          if(L[x][y]==i+1):
            black[x][y] = 255
      img2, contours, hierarchy = cv2.findContours(black,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
      black = 255 - black
      contour = contours[0]
      [x,y,w,h] = cv2.boundingRect(contour)

      #Don't plot small false positives that aren't text
      if w > 70 and h > 70 :
        cv2.drawContours(final, contours, -1, (0,255,0), 3)
        continue

      if w < 15 and h < 15 :
        continue

      cropped = np.zeros((h + 6, w + 6, 1), np.uint8)
      cropped[:] = 255

      for p in range(3, h + 3):
        for q in range(3, w + 3):
          if(black[y + p - 3][x + q - 3] != 255 ):
            cropped[p][q] = 0

      cv2.imwrite('crop'+str(i+1)+'.jpg', cropped)
      c = my_recognizer('crop'+str(i+1))

      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(final, c ,(x + w/2, y + h/2), font, 1,(0,255,0),1,cv2.LINE_AA)

    # write original image with added contours to disk
    cv2.imshow('captcha_result' , img)
    cv2.waitKey()

    cv2.imshow('text', final)
    cv2.waitKey()
    cv2.imwrite('detected_char.jpg', final)


file_name ='8.jpg'
captch_ex(file_name)