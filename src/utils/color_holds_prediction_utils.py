#! /usr/bin/python

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

# RGB colors for drawing bounding boxes
color_dict_rgb = {'blue': (20, 20, 255), 
				'black': (25, 25, 25), 
				'green': (0, 255, 0), 
				'red': (255, 20, 20), 
				'white': (255,255,255),
				'pink': (255,110,210),
				'purple': (190,110,255),
				'yellow': (255,255,0),
				'orange': (255,128,0),
				'gray': (160,160,160),
				'darkgray':(70,70,70)
				} 

color_dict_HSV = {'black': [[180, 255, 39], [0, 0, 0]],
			  'white': [[180, 18, 255], [0, 0, 231]],
			  'pink' : [[169, 255, 255], [160, 50, 45]],
			  'red1': [[180, 255, 255], [170, 50, 45]],
			  'red2': [[9, 255, 255], [0, 50, 45]],
			  'green': [[86, 255, 255], [36, 50, 45]],
			  'blue': [[128, 255, 255], [87, 50, 45]],
			  'yellow': [[35, 255, 255], [16, 50, 45]],
			  'purple': [[159, 255, 255], [129, 50, 45]],
			  'orange': [[15, 255, 255], [10, 50, 45]],
			  'gray': [[180, 18, 230], [0, 0, 40]]}

#color_dict_HSV = {'black': [[180, 255, 45], [0, 0, 0]],
#              'white': [[180, 18, 255], [0, 0, 230]],
#              'pink': [[180,255,255], [146,50,50]], 
#              'red1': [[180, 255, 255], [146, 50, 70]],
#              'red2': [[10, 255, 255], [0, 30, 70]],
#              'green': [[89, 255, 255], [36, 50, 60]],
#              'blue': [[126, 255, 255], [90, 50, 70]],
#              'yellow': [[35, 255, 255], [20, 50, 70]], 
#              'purple': [[145, 255, 255], [127, 50, 70]],
#              'orange': [[20, 255, 255], [10, 50, 70]],
#              'gray': [[180, 18, 230], [0, 0, 40]]}

def kmean_centers(img):
	#plt.imshow(img)
	plt.show()
	data = np.reshape(img, (-1,3))
	#print(data.shape)
	data = np.float32(data)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	compactness,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags)

	#print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
	#print(centers)
	return centers


def getholdcolor(centers):
	colors = []
	for i in range(len(centers)):
		h,s,v = centers[i]
		for color in color_dict_HSV:
			#print(color_dict_HSV[color])
			hi_h, hi_s, hi_v = color_dict_HSV[color][0]
			lo_h, lo_s, lo_v = color_dict_HSV[color][1]
			if lo_h-1 <= h and h <= hi_h+1:
				if lo_s-1 <= s and s <= hi_s+1:
					if lo_v-1 <= v and v <= hi_v+1:
						colors.append(color)
						#print(color, " ", centers[i])

	if len(colors) == 0:
		return 'gray'
	if len(colors) == 1:
		return colors[0]    
	if  len(colors) > 1 and (('red1' in colors and 'red2' in colors) or ('pink' in colors and 'red2' in colors)):
		return 'red'
	elif colors[0] == 'gray':
		if colors[1] == 'gray' and len(colors) > 2:
			return colors[2]
		else:
			return colors[1]
	else:
		return colors[0]


def getAllHoldColors(fn, holds):
	# open image, convert
	img = cv2.imread(fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	colors = []
	for h in holds:
		x_low, y_low = h[0]
		x_high, y_high = h[1]
		centers = kmean_centers(img[y_low:y_high, x_low:x_high,:])
		color = getholdcolor(centers)
		if color == 'red1' or color =='red2':
			colors.append('red')
		else:
			colors.append(color)
	return holds, colors


def draw_bounds(holds, colors, rgb_img):
	"""
	Draws the bounding boxes on an image given list of holds and list of colors
	"""
	img_cp = rgb_img.copy()
	for i in range(len(holds)):
		h = holds[i]
		x_low, y_low = h[0]
		x_high, y_high = h[1]
		cv2.rectangle(img_cp,(x_low, y_low),(x_high, y_high),color_dict_rgb[colors[i]],2)
	plt.imshow(img_cp)
	plt.show()
	return img_cp

