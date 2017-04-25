# For Capturing warnings on terminal
import logging        
logging.captureWarnings(True)

# For command line execution
import os           

# Importing various libraries of flask
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, Markup
from werkzeug import secure_filename

# Import OpenCV python libraries
import cv2

# Other libraries for Image Processing
from PIL import Image
import sys
import math
import imutils
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle


from skimage import measure
from OneWordSolve import my_recognizer



# Initialize the Flask application
app = Flask(__name__, static_url_path = "", static_folder = "static")

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    return dist

def centroid_func(point_list):

    centroid_p = [0.0, 0.0]
    signedArea = 0.0
    x0 = 0.0
    y0 = 0.0
    x1 = 0.0
    y1 = 0.0
    a = 0.0

    for i in range(len(point_list)-1):
        x0 = point_list[i][0]
        y0 = point_list[i][1]
        x1 = point_list[i+1][0]
        y1 = point_list[i+1][1]
        a = x0*y1 - x1*y0
        signedArea = signedArea + a
        centroid_p[0] = centroid_p[0] + (x0 + x1)*a
        centroid_p[1] = centroid_p[1] + (y0 + y1)*a

    signedArea = signedArea*(0.5)
    centroid_p[0] = centroid_p[0]/(6.0*signedArea)
    centroid_p[1] = centroid_p[1]/(6.0*signedArea)

    return centroid_p


def get_lines(contours):
    white = np.zeros((478,850,1), np.uint8)
    white[:] = 255

    contour_size = len(contours)

    line_list = []

    for i in range(contour_size):
        
        (x_c,y_c),radius = cv2.minEnclosingCircle(contours[i])
        x_c = int(x_c)
        y_c = int(y_c)
        center = (x_c, y_c)
        radius = int(radius)
        if(2*radius < 4):
			continue


        rows,cols = white.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(contours[i], cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        x0 = cols-1
        y0 = righty
        x1 = 0
        y1 = lefty

        if(2*radius > 4):
	        cv2.line(white,(x0,y0),(x1,y1),(0,255,0),1)
	        cv2.drawContours(white, contours, i , (0,255,0), 1)
	        cv2.circle(white,center,radius,(0,255,0),1)

        alpha = (float(y0 - y1))/x0
        beta = y1 - y_c

        a = 1 + alpha*alpha
        b = 2*alpha*beta - 2*x_c
        c = x_c*x_c + beta*beta - radius*radius

        try :
        	line_x1 =  (- b + math.sqrt(b*b - 4*a*c))/(2*a)
        	line_x2 =  (- b - math.sqrt(b*b - 4*a*c))/(2*a)
        	line_y1 = alpha*line_x1 + y1
        	line_y2 = alpha*line_x2 + y1
        except :
			continue

        if(distance(line_x1, line_y1, line_x2, line_y2) > 7):
            cv2.line(white,(int(line_x1), int(line_y1)),(int(line_x2),int(line_y2)),(0,255,0),3)
            a_line = [line_x1, line_y1, line_x2, line_y2]
            line_list.append(a_line)

    #print line_list
    cv2.imwrite("lines.jpg", white)
    # cv2.imshow("Lines", white)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return line_list

def recur_shapes(line_list, single_list, my_bool):
	empty = []
	match = single_list[-1]
	extra = single_list[-2]
	check = False
	for i in range(0,len(line_list)):
		white = np.zeros((478,850,1), np.uint8)
		white[:] = 255
		list1 = [line_list[i][0], line_list[i][1]]
		list2 = [line_list[i][2], line_list[i][3]]
		if(match==list1 and extra!=list2):
			if(my_bool[i]==False):
				my_bool[i] = True
				single_list.append(list2)
				if(single_list[-1]==single_list[0]):
					print 'Shape : ', len(single_list)-1
					return True, single_list
				check, list_till = recur_shapes(line_list, single_list, my_bool)
				if(check==True):
					return True, list_till
				else:
					single_list.pop()		
			else:
				return False, empty
				
		
		elif(match==list2 and extra!=list1):
			if(my_bool[i]==False):
				print '$$'
				my_bool[i] = True
				single_list.append(list1)
				if(single_list[-1]==single_list[0]):
					print 'Shape : ', len(single_list)-1
					return True, single_list
				check, list_till = recur_shapes(line_list, single_list, my_bool)
				if(check==True):
					return True, list_till
				else:
					single_list.pop()
			else:
				return False, empty
				print '##'
				

	print '@@'
	return False, empty


def get_shapes(line_list):
	shape_list = []
	line_count = len(line_list)
	count = 0
	while(len(line_list)!=0 and count != line_count):
		count = count + 1
		my_bool = [False for i in range(1000)]
		single_list = [[line_list[0][0], line_list[0][1]], [line_list[0][2], line_list[0][3]]]
		my_bool[0] = True
		check, poly_list = recur_shapes(line_list, single_list, my_bool)
		if(check==True):
			shape_list.append(poly_list)
			for i in range(len(poly_list)-1):
				delete_line_1 = [poly_list[i][0], poly_list[i][1], poly_list[i+1][0], poly_list[i+1][1]]
				delete_line_2 = [poly_list[i+1][0], poly_list[i+1][1], poly_list[i][0], poly_list[i][1]]
				if delete_line_1 in line_list:
					line_list.remove(delete_line_1)
				elif delete_line_2 in line_list:
					line_list.remove(delete_line_2)
		else:
			temp = line_list.pop(0)
			line_list.append(temp)

	return shape_list, line_list


def draw_rect(white, point_list):

	x_min = min(point_list[0][0], point_list[1][0], point_list[2][0], point_list[3][0])
	x_max = max(point_list[0][0], point_list[1][0], point_list[2][0], point_list[3][0])
	y_min = min(point_list[0][1], point_list[1][1], point_list[2][1], point_list[3][1])
	y_max = max(point_list[0][1], point_list[1][1], point_list[2][1], point_list[3][1])

	cv2.line(white, (int(x_min), int(y_min)), (int(x_max), int(y_min) ),(0,255,0),2)
	cv2.line(white, (int(x_min), int(y_max)), (int(x_max), int(y_max) ),(0,255,0),2)
	cv2.line(white, (int(x_min), int(y_min)), (int(x_min), int(y_max) ),(0,255,0),2)
	cv2.line(white, (int(x_max), int(y_min)), (int(x_max), int(y_max) ),(0,255,0),2)

	font = cv2.FONT_HERSHEY_SIMPLEX
	# cv2.putText(white,'Rectangle',(int((x_min + x_max)/2) - 30,int((y_min + y_max)/2)), font, 0.5,(0,255,0),1,cv2.LINE_AA)

	return white

def draw_poly(white, point_list, line_list):
    n = len(point_list) - 1
    centroid = centroid_func(point_list)
    dist = 0.0
    for i in range(n):
        dist = dist + distance(centroid[0], centroid[1], point_list[i][0], point_list[i][1])
    
    dist = dist/n

    p_list = [[centroid[0], centroid[1] - dist]]


    R = dist
    a = math.acos(0)

    for i in range(1,n):
        p = [centroid[0] + R*(math.cos(a + 2*(math.pi)*i/n)), centroid[1] - R*(math.sin(a + 2*(math.pi)*i/n))]
        p_list.append(p)

    for i in range(n):
        if(i==n-1):
            cv2.line(white, (int(p_list[i][0]), int(p_list[i][1])), (int(p_list[0][0]), int(p_list[0][1]) ),(0,255,0),2)
        else:
            cv2.line(white, (int(p_list[i][0]), int(p_list[i][1])), (int(p_list[i+1][0]), int(p_list[i+1][1]) ),(0,255,0),2)

    return_list = []

    if(n==4):
    	for i in range(n):
    		print "Hello"
    		for j in range(len(line_list)):
    			if((line_list[j][0], line_list[j][1])==(point_list[i][0],point_list[i][1])):
    				print "OOPS"
    				p1 = distance(line_list[j][0], line_list[j][1], p_list[0][0], p_list[0][1])
    				p2 = distance(line_list[j][0], line_list[j][1], p_list[1][0], p_list[1][1])
    				p3 = distance(line_list[j][0], line_list[j][1], p_list[2][0], p_list[2][1])
    				p4 = distance(line_list[j][0], line_list[j][1], p_list[3][0], p_list[3][1])
    				min_dist = min(p1, p2, p3, p4)
    				if(min_dist==p1):
    					temp = [j, p_list[0][0], p_list[0][1], line_list[j][2], line_list[j][3]]
    					return_list.append(temp)
    				if(min_dist==p2):
    					temp = [j, p_list[1][0], p_list[1][1], line_list[j][2], line_list[j][3]]
    					return_list.append(temp)
    				if(min_dist==p3):
    					temp = [j, p_list[2][0], p_list[2][1], line_list[j][2], line_list[j][3]]
    					return_list.append(temp)
    				if(min_dist==p4):
    					temp = [j, p_list[3][0], p_list[3][1], line_list[j][2], line_list[j][3]]
    					return_list.append(temp)
    			elif((line_list[j][2], line_list[j][3])==(point_list[i][0], point_list[i][1])):
    				print "OOPS"
    				p1 = distance(line_list[j][2], line_list[j][3], p_list[0][0], p_list[0][1])
    				p2 = distance(line_list[j][2], line_list[j][3], p_list[1][0], p_list[1][1])
    				p3 = distance(line_list[j][2], line_list[j][3], p_list[2][0], p_list[2][1])
    				p4 = distance(line_list[j][2], line_list[j][3], p_list[3][0], p_list[3][1])
    				min_dist = min(p1, p2, p3, p4)
    				if(min_dist==p1):
    					temp = [j, line_list[j][0], line_list[j][1], p_list[0][0], p_list[0][1]]
    					return_list.append(temp)
    				if(min_dist==p2):
    					temp = [j, line_list[j][0], line_list[j][1], p_list[1][0], p_list[1][1]]
    					return_list.append(temp)
    				if(min_dist==p3):
    					temp = [j, line_list[j][0], line_list[j][1], p_list[2][0], p_list[2][1]]
    					return_list.append(temp)
    				if(min_dist==p4):
    					temp = [j, line_list[j][0], line_list[j][1], p_list[3][0], p_list[3][1]]
    					return_list.append(temp)

    	return white, return_list

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # if(n==3):
    #     cv2.putText(white,'Triangle',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==4):
    #     cv2.putText(white,'Rhombus',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==5):
    #     cv2.putText(white,'Pentagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==6):
    #     cv2.putText(white,'Hexagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==7):
    #     cv2.putText(white,'Septagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==8):
    #     cv2.putText(white,'Octagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==9):
    #     cv2.putText(white,'Nonagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==10):
    #     cv2.putText(white,'Decagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==11):
    #     cv2.putText(white,'Undecagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    # if(n==12):
    #     cv2.putText(white,'Dodecagon',(int(centroid[0]) - 30,int(centroid[1])), font, 0.5,(0,255,0),1,cv2.LINE_AA)

    return white, return_list

def draw_line(white, point_list):

    cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[1][0]), int(point_list[1][1]) ),(0,255,0),2)
    
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(white,'Line',(int((point_list[0][0] + point_list[1][0])/2) - 20,int((point_list[0][1] + point_list[1][1])/2) - 10), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    
    return white

def draw_parallelogram(white, point_list):
	d1 = distance(point_list[0][0], point_list[0][1], point_list[2][0], point_list[2][1])
	d2 = distance(point_list[1][0], point_list[1][1], point_list[3][0], point_list[3][1])

	slope_1 = (point_list[2][1] - point_list[0][1])/(point_list[2][0] - point_list[0][0])
	slope_2 = (point_list[3][1] - point_list[1][1])/(point_list[3][0] - point_list[1][0])

	angle_1 = math.degrees(math.atan(slope_1))
	angle_2 = math.degrees(math.atan(slope_2))

	font = cv2.FONT_HERSHEY_SIMPLEX
	
	if(d1 > d2):
		if(angle_1 > 0):
			if(point_list[0][1] < point_list[2][1]):
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[2][0] + d1/4), int(point_list[0][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[0][0] - d1/4), int(point_list[2][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[2][0] + d1/4), int(point_list[0][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[0][0] - d1/4), int(point_list[2][1]) ),(0,255,0),2)
			else:
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[2][0] - d1/4), int(point_list[0][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[0][0] + d1/4), int(point_list[2][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[2][0] - d1/4), int(point_list[0][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[0][0] + d1/4), int(point_list[2][1]) ),(0,255,0),2)
		else:
			if(point_list[0][1] < point_list[2][1]):
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[2][0] - d1/4), int(point_list[0][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[0][0] + d1/4), int(point_list[2][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[2][0] - d1/4), int(point_list[0][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[0][0] + d1/4), int(point_list[2][1]) ),(0,255,0),2)
			else:
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[2][0] + d1/4), int(point_list[0][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[0][0]), int(point_list[0][1])), (int(point_list[0][0] - d1/4), int(point_list[2][1]) ),(0,255,0),1)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[2][0] + d1/4), int(point_list[0][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[2][0]), int(point_list[2][1])), (int(point_list[0][0] - d1/4), int(point_list[2][1]) ),(0,255,0),2)

		# cv2.putText(white,'Parallelogram',(int((point_list[0][0] + point_list[2][0])/2) - 30,int((point_list[0][1] + point_list[2][1])/2)), font, 0.5,(0,255,0),1,cv2.LINE_AA)
	
	else:
		if(angle_2 > 0):
			if(point_list[1][1] < point_list[3][1]):
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[3][0] - d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[1][0] + d2/4), int(point_list[3][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[3][0] - d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[1][0] + d2/4), int(point_list[3][1]) ),(0,255,0),2)
			else:
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[3][0] + d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[1][0] - d2/4), int(point_list[3][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[3][0] + d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[1][0] - d2/4), int(point_list[3][1]) ),(0,255,0),2)
		else:
			if(point_list[1][1] < point_list[3][1]):
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[3][0] + d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[1][0] - d2/4), int(point_list[3][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[3][0] + d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[1][0] - d2/4), int(point_list[3][1]) ),(0,255,0),2)
			else:
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[3][0] - d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[1][0]), int(point_list[1][1])), (int(point_list[1][0] + d2/4), int(point_list[3][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[3][0] - d2/4), int(point_list[1][1]) ),(0,255,0),2)
				cv2.line(white, (int(point_list[3][0]), int(point_list[3][1])), (int(point_list[1][0] + d2/4), int(point_list[3][1]) ),(0,255,0),2)

		# cv2.putText(white,'Parallelogram',(int((point_list[1][0] + point_list[3][0])/2) - 30,int((point_list[1][1] + point_list[3][1])/2)), font, 0.5,(0,255,0),1,cv2.LINE_AA)

	return white


def check_rhombus(point_list):

	if((point_list[2][0] == point_list[0][0]) or (point_list[3][0] == point_list[1][0])):
		return 1
	
	slope_1 = (point_list[2][1] - point_list[0][1])/(point_list[2][0] - point_list[0][0])
	slope_2 = (point_list[3][1] - point_list[1][1])/(point_list[3][0] - point_list[1][0])

	angle_1 = math.degrees(math.atan(slope_1))
	angle_2 = math.degrees(math.atan(slope_2))

	print angle_1
	print angle_2

	if(angle_1 > 70.0 or angle_1 < -70.0 or angle_2 > 70.0 or angle_2 < -70.0):
		return 1

	return 0

def check_parallelogram(point_list):
	d1 = distance(point_list[0][0], point_list[0][1], point_list[2][0], point_list[2][1])
	d2 = distance(point_list[1][0], point_list[1][1], point_list[3][0], point_list[3][1])

	if(d1 > d2):
		if(d1 > 1.2*d2):
			return 1
	if(d2 > d1):
		if(d2 > 1.2*d1):
			return 1
	return 0

def char_recognizer(img):

	img = imutils.resize(img, width=850)
	crop_image = img
	img_final = img

	white = np.zeros((478,850,1), np.uint8)
	white[:] = 255

	'''
	    line  8 to 12  : Remove noisy portion 
	'''
	bin_image = cv2.adaptiveThreshold(img_final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,89,18)

	# cv2.imwrite('bin.jpg', bin_image)

	result = 255-bin_image

	kernel = np.ones((2,2),np.uint8)
	result = cv2.dilate(result,kernel,iterations = 1)

	height, width = result.shape[:2]

	# cv2.imshow('captcha_result' , result)
	# cv2.waitKey()

	L = measure.label(result, background = 0)
	print "Number of components:", np.max(L)

	count = np.max(L)

	final = np.zeros((478,850,1), np.uint8)
	final[:] = 255

	second = np.zeros((478,850,1), np.uint8)
	second[:] = 255

	for i in range(count):
		black = np.zeros((478,850,1), np.uint8)
		for x in range(height):
			for y in range(width):
	  			if(L[x][y]==i+1):
	  				black[x][y] = 255
		img2, contours, hierarchy = cv2.findContours(black,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
		black = 255 - black
		if not contours:
			continue
		contour = contours[0]
		[x,y,w,h] = cv2.boundingRect(contour)

		#Don't plot small false positives that aren't text
		if w > 40 and h > 40 :
			# cv2.drawContours(final, contours, -1, (0,255,0), 3)
			continue

		if w < 7 and h < 7 :
			continue

		cropped = np.zeros((h + 6, w + 6, 1), np.uint8)
		cropped[:] = 255

		for p in range(3, h + 3):
			for q in range(3, w + 3):
		  		if(black[y + p - 3][x + q - 3] != 255 ):
		  			cropped[p][q] = 0

		cv2.drawContours(result, contours, -1, (0,255,0), 3)
		# for p in range(y, y+h):
		# 	for q in range(x, x+w):
		# 		result[p][q] = 0

		cv2.imwrite('crop'+str(i+1)+'.jpg', cropped)
		c = my_recognizer('crop'+str(i+1))

		if(c==0):
			continue

		font = cv2.FONT_HERSHEY_SIMPLEX
		if c[0]=='U':
			cv2.putText(final, 'u' ,(x, y + h), font, 0.7	,(0,255,0),1,cv2.LINE_AA)
			cv2.putText(second, 'u' ,(x, y + h), font, 0.7 ,(0,255,0),1,cv2.LINE_AA)
		elif c[0]=='g' or c[0]=='j' or c[0]=='q' or c[0]=='p' or c[0]=='y':
			cv2.putText(final, c ,(x, y + h/2), font, 0.7	,(0,255,0),1,cv2.LINE_AA)
			cv2.putText(second, c ,(x, y + h/2), font, 0.7 ,(0,255,0),1,cv2.LINE_AA)
		else:
			cv2.putText(final, c ,(x, y + h), font, 0.7	,(0,255,0),1,cv2.LINE_AA)
			cv2.putText(second, c ,(x, y + h), font, 0.7 ,(0,255,0),1,cv2.LINE_AA)

	# write original image with added contours to disk
	# cv2.imshow('captcha_result' , img)
	# cv2.waitKey()

	# cv2.imshow('text', final)
	# cv2.waitKey()
	cv2.imwrite('detected_char.jpg', final)
	return second, final, result


# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)

        # Creating the new folder and deleting existing one with same name
        if os.path.exists('static/'+file.filename):
            os.rmdir('static/'+file.filename)

        os.makedirs('static/'+file.filename)


        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        
        image = cv2.imread('uploads/'+file.filename,0)

        # Saving input image in above folder
        cv2.imwrite('static/'+file.filename+'/input.jpg',imutils.resize(image, width=850))

        final_image, white, result = char_recognizer(image)

        cv2.imwrite('static/'+file.filename+'/char_removed.jpg',result)

     #    bin_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,89,18)
     #    result = 255-bin_image
     #    kernel = np.ones((9,9),np.uint8)
     #    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

     #    new_kernel = np.ones((2,2),np.uint8)
    	# result = cv2.dilate(result,new_kernel,iterations = 1)

     #    result = imutils.resize(result, width=850)
     #    ret, result = cv2.threshold(result,127,255,cv2.THRESH_BINARY)

        dst = cv2.cornerHarris(result,6,5,0.07)
        result[ dst > 0.01 * dst.max() ]=[0]

        # Saving input image in above folder
        cv2.imwrite('static/'+file.filename+'/corner_image.jpg',result)

        im2, contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        line_list = get_lines(contours)

        print '############'
        print len(line_list)
        print '############'

        only_lines = np.zeros((478,850,1), np.uint8)
        only_lines[:] = 255

        for i in range(len(line_list)):
        	cv2.line(only_lines, (int(line_list[i][0]), int(line_list[i][1])), (int(line_list[i][2]), int(line_list[i][3]) ),(0,255,0),2)

		cv2.imwrite('only_lines.jpg', only_lines)

        bool_arr = np.zeros((1, 1000))

        # Join 2 straight lines disconnected due to corner detection

        for i in range(len(line_list)):
            if(bool_arr[0][i] == 0):
                for j in range(i+1, len(line_list)):
                    if(i!=j and bool_arr[0][j] == 0):
                        p = -1
                        if(distance(line_list[i][0], line_list[i][1], line_list[j][0], line_list[j][1]) < 15):
                            p = 0
                        if(distance(line_list[i][0], line_list[i][1], line_list[j][2], line_list[j][3]) < 15):
                            p = 1
                        if(distance(line_list[i][2], line_list[i][3], line_list[j][0], line_list[j][1]) < 15):
                            p = 2
                        if(distance(line_list[i][2], line_list[i][3], line_list[j][2], line_list[j][3]) < 15):
                            p = 3

                        if(p == -1):
                            continue

                        angle_1 = 0.0
                        angle_2 = 0.0

                        if(line_list[i][0] == line_list[i][2]):
                            angle_1 = 90.0
                        if(line_list[j][0] == line_list[j][2]):
                            angle_2 = 90.0

                        if(angle_1 == 0.0):
                            slope_1 = (line_list[i][1] - line_list[i][3])/(line_list[i][0] - line_list[i][2])
                            angle_1 = math.degrees(math.atan(slope_1))
                        if(angle_2 == 0.0):
                            slope_2 = (line_list[j][1] - line_list[j][3])/(line_list[j][0] - line_list[j][2])
                            angle_2 = math.degrees(math.atan(slope_2))

                        if(angle_1 < 0.0):
                            angle_1 = 180 + angle_1
                        if(angle_2 < 0.0):
                            angle_2 = 180 + angle_2

                        if((angle_2 - angle_1) < 20.0 and (angle_2 - angle_1) > -20.0):
                            if(p==0):
                                line_list[i][0] = line_list[j][2]
                                line_list[i][1] = line_list[j][3]
                            if(p==1):
                                line_list[i][0] = line_list[j][0]
                                line_list[i][1] = line_list[j][1]
                            if(p==2):
                                line_list[i][2] = line_list[j][2]
                                line_list[i][3] = line_list[j][3]
                            if(p==3):
                                line_list[i][2] = line_list[j][0]
                                line_list[i][3] = line_list[j][1]
                            
                            print '****'
                            bool_arr[0][j] = 1
                            break

        x = len(line_list) - 1

        while(x>=0):
            if(bool_arr[0][x]==1):
                line_list.pop(x)
            x = x-1

        print 'Number of lines : ' , len(line_list)

        merge = np.zeros((478,850,1), np.uint8)
        merge[:] = 255

        for i in range(len(line_list)):
        	cv2.line(merge, (int(line_list[i][0]), int(line_list[i][1])), (int(line_list[i][2]), int(line_list[i][3]) ),(0,255,0),2)

		cv2.imwrite('merge.jpg', merge)

        # Join together 3 concurrent lines

        concurrent_points = []

        for i in range(len(line_list)):
        	for j in range(i+1, len(line_list)):
        		for k in range(j+1, len(line_list)):
        			if(i != j and i != k):
        				if(distance(line_list[i][0], line_list[i][1], line_list[j][0], line_list[j][1]) < 15 and distance(line_list[i][0], line_list[i][1], line_list[k][0], line_list[k][1]) < 15):
        					
        					print 'case 1'

        					line_list[j][0] = line_list[i][0]
        					line_list[j][1] = line_list[i][1]

        					line_list[k][0] = line_list[i][0]
        					line_list[k][1] = line_list[i][1]

        					concurrent_points.append((line_list[i][0], line_list[i][1]))

        					continue

        				if(distance(line_list[i][0], line_list[i][1], line_list[j][0], line_list[j][1]) < 15 and distance(line_list[i][0], line_list[i][1], line_list[k][2], line_list[k][3]) < 15):
        					centroid_x = (line_list[i][0] + line_list[j][0] + line_list[k][2])/3 
        					centroid_y = (line_list[i][1] + line_list[j][1] + line_list[k][3])/3

        					print 'case 2'

        					line_list[i][0] = centroid_x
        					line_list[i][1] = centroid_y

        					line_list[j][0] = centroid_x
        					line_list[j][1] = centroid_y

        					line_list[k][2] = centroid_x
        					line_list[k][3] = centroid_y

        					concurrent_points.append((centroid_x, centroid_y))

        					continue

        				if(distance(line_list[i][0], line_list[i][1], line_list[j][2], line_list[j][3]) < 15 and distance(line_list[i][0], line_list[i][1], line_list[k][0], line_list[k][1]) < 15):
        					centroid_x = (line_list[i][0] + line_list[j][2] + line_list[k][0])/3 
        					centroid_y = (line_list[i][1] + line_list[j][3] + line_list[k][1])/3

        					print 'case 3'

        					line_list[i][0] = centroid_x
        					line_list[i][1] = centroid_y

        					line_list[j][2] = centroid_x
        					line_list[j][3] = centroid_y

        					line_list[k][0] = centroid_x
        					line_list[k][1] = centroid_y

        					concurrent_points.append((centroid_x, centroid_y))

        					continue

        				if(distance(line_list[i][0], line_list[i][1], line_list[j][2], line_list[j][3]) < 15 and distance(line_list[i][0], line_list[i][1], line_list[k][2], line_list[k][3]) < 15):
        					centroid_x = (line_list[i][0] + line_list[j][2] + line_list[k][2])/3 
        					centroid_y = (line_list[i][1] + line_list[j][3] + line_list[k][3])/3

        					print 'case 4'

        					line_list[j][2] = line_list[i][0]
        					line_list[j][3] = line_list[i][1]

        					line_list[k][2] = line_list[i][0]
        					line_list[k][3] = line_list[i][1]

        					concurrent_points.append((line_list[i][0], line_list[i][1]))

        					continue

        				if(distance(line_list[i][2], line_list[i][3], line_list[j][0], line_list[j][1]) < 15 and distance(line_list[i][2], line_list[i][3], line_list[k][0], line_list[k][1]) < 15):
        					centroid_x = (line_list[i][2] + line_list[j][0] + line_list[k][0])/3 
        					centroid_y = (line_list[i][3] + line_list[j][1] + line_list[k][1])/3

        					print 'case 5'

        					line_list[j][0] = line_list[i][2]
        					line_list[j][1] = line_list[i][3]
	
	       					line_list[k][0] = line_list[i][2]
        					line_list[k][1] = line_list[i][3]

        					concurrent_points.append((line_list[i][2], line_list[i][3]))

        					continue

        				if(distance(line_list[i][2], line_list[i][3], line_list[j][0], line_list[j][1]) < 15 and distance(line_list[i][2], line_list[i][3], line_list[k][2], line_list[k][3]) < 15):
        					centroid_x = (line_list[i][2] + line_list[j][0] + line_list[k][2])/3 
        					centroid_y = (line_list[i][3] + line_list[j][1] + line_list[k][3])/3

        					print 'case 6'

        					line_list[i][2] = centroid_x
        					line_list[i][3] = centroid_y

        					line_list[j][0] = centroid_x
        					line_list[j][1] = centroid_y

        					line_list[k][2] = centroid_x
        					line_list[k][3] = centroid_y

        					concurrent_points.append((centroid_x, centroid_y))

        					continue

        				if(distance(line_list[i][2], line_list[i][3], line_list[j][2], line_list[j][3]) < 15 and distance(line_list[i][2], line_list[i][3], line_list[k][0], line_list[k][1]) < 15):
        					centroid_x = (line_list[i][2] + line_list[j][2] + line_list[k][0])/3 
        					centroid_y = (line_list[i][3] + line_list[j][3] + line_list[k][1])/3

        					print 'case 7'

        					line_list[i][2] = centroid_x
        					line_list[i][3] = centroid_y

        					line_list[j][2] = centroid_x
        					line_list[j][3] = centroid_y

        					line_list[k][0] = centroid_x
        					line_list[k][1] = centroid_y

        					concurrent_points.append((centroid_x, centroid_y))

        					continue

        				if(distance(line_list[i][2], line_list[i][3], line_list[j][2], line_list[j][3]) < 15 and distance(line_list[i][2], line_list[i][3], line_list[k][2], line_list[k][3]) < 15):
        					centroid_x = (line_list[i][2] + line_list[j][2] + line_list[k][2])/3 
        					centroid_y = (line_list[i][3] + line_list[j][3] + line_list[k][3])/3

        					print 'case 8'

        					line_list[j][2] = line_list[i][2]
        					line_list[j][3] = line_list[i][3]

           					line_list[k][2] = line_list[i][2]
        					line_list[k][2] = line_list[i][3]

        					concurrent_points.append((line_list[i][2], line_list[i][3]))


        conc = np.zeros((478,850,1), np.uint8)
        conc[:] = 255

        for i in range(len(line_list)):
        	cv2.line(conc, (int(line_list[i][0]), int(line_list[i][1])), (int(line_list[i][2]), int(line_list[i][3]) ),(0,255,0),2)

		cv2.imwrite('concurrent.jpg', conc)

        # Merging line intersection

        for i in range(len(line_list)):
            for j in range(len(line_list)):
                if(i != j):
                    a1 = line_list[i][1] - line_list[i][3]
                    b1 = line_list[i][2] - line_list[i][0] 
                    c1 = line_list[i][2]*line_list[i][1] - line_list[i][0]*line_list[i][3]

                    a2 = line_list[j][1] - line_list[j][3]
                    b2 = line_list[j][2] - line_list[j][0]
                    c2 = line_list[j][2]*line_list[j][1] - line_list[j][0]*line_list[j][3]

                    if(a1*b2 == a2*b1):
                        continue
                    x_p = (c1*b2 - c2*b1)/(a1*b2 - a2*b1)
                    y_p = (c1*a2 - c2*a1)/(a2*b1 - a1*b2)
                    p1 = -1
                    p2 = -1
                    if(distance(x_p, y_p, line_list[i][0], line_list[i][1]) < distance(x_p, y_p, line_list[i][2], line_list[i][3])):
                        p1 = 0
                    else:
                        p1 = 2

                    if(distance(x_p, y_p, line_list[j][0], line_list[j][1]) < distance(x_p, y_p, line_list[j][2], line_list[j][3])):
                        p2 = 0
                    else:
                        p2 = 2

                    if(distance(x_p, y_p, line_list[i][p1], line_list[i][p1+1]) < 15 and distance(x_p, y_p, line_list[j][p2], line_list[j][p2+1]) < 15 or distance(line_list[i][p1], line_list[i][p1+1], line_list[j][p2], line_list[j][p2+1]) < 15):
                        line_list[i][p1] = x_p
                        line_list[i][p1+1] = y_p
                        line_list[j][p2] = x_p
                        line_list[j][p2+1] = y_p


        for i in range(len(line_list)):
            cv2.line(white,(int(line_list[i][0]), int(line_list[i][1])),(int(line_list[i][2]),int(line_list[i][3])),(0,255,0),2)

        for i in range(len(line_list)):
        	line_list[i][0] = int(line_list[i][0])
        	line_list[i][1] = int(line_list[i][1])
        	line_list[i][2] = int(line_list[i][2])
        	line_list[i][3] = int(line_list[i][3])

        box = np.zeros((478,850,1), np.uint8)
        box[:] = 255

        for i in range(len(line_list)):
        	cv2.line(box, (int(line_list[i][0]), int(line_list[i][1])), (int(line_list[i][2]), int(line_list[i][3]) ),(0,255,0),2)

		cv2.imwrite('box.jpg', box)

        # Saving input image in above folder
        cv2.imwrite('static/'+file.filename+'/closed_image.jpg',white)

        shape_list, plain_lines = get_shapes(line_list)

        # for i in range(len(shape_list)):
        #   print len(shape_list[i])

        perfect_shape = final_image

   #      for i in range(len(plain_lines)):
			# perfect_shape = draw_line(perfect_shape, ((plain_lines[i][0], plain_lines[i][1]), (plain_lines[i][2], plain_lines[i][3])))

        for i in range(len(shape_list)):
            if(len(shape_list[i])==5):
                if(check_rhombus(shape_list[i])==1):
                    perfect_shape, change_lines = draw_poly(perfect_shape, shape_list[i], plain_lines)
                    # for j in range(len(change_lines)):
                    # 	print 'Hehahaha'
                    # 	plain_lines[change_lines[j][0]][0] = change_lines[j][1]
                    # 	plain_lines[change_lines[j][0]][1] = change_lines[j][2]
                    # 	plain_lines[change_lines[j][0]][2] = change_lines[j][3]
                    # 	plain_lines[change_lines[j][0]][3] = change_lines[j][4]

                elif(check_parallelogram(shape_list[i])==1):
                	perfect_shape = draw_parallelogram(perfect_shape, shape_list[i])
                else:
                    perfect_shape = draw_rect(perfect_shape, shape_list[i])
            elif(len(shape_list[i])!=2):
                perfect_shape, change_lines = draw_poly(perfect_shape, shape_list[i], plain_lines)

        for i in range(len(plain_lines)):
			perfect_shape = draw_line(perfect_shape, ((plain_lines[i][0], plain_lines[i][1]), (plain_lines[i][2], plain_lines[i][3])))

		

        cv2.imwrite('static/'+file.filename+'/final_image.jpg',perfect_shape)

        # Creating Markups for index1.html
        input_image = Markup('<img src="/'+file.filename+'/input.jpg" alt="Final Image" style="width:504px;height:280px;">')
        corner_image = Markup('<img src="/'+file.filename+'/corner_image.jpg" alt="Final Image" style="width:504px;height:280px;">')
        closed_image = Markup('<img src="/'+file.filename+'/closed_image.jpg" alt="Final Image" style="width:504px;height:280px;">')
        final_image = Markup('<img src="/'+file.filename+'/final_image.jpg" alt="Final Image" style="width:504px;height:280px;">')

        # Calling index1.html file with Markups as parameters
        return render_template('index1.html', input_image=input_image, corner_image=corner_image, closed_image=closed_image, final_image=final_image)

    else:
        return render_template('index2.html')

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(
        host="127.0.0.3",
        port=int("5500"),
        debug=True
    )