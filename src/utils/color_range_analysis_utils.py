import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# HSV color ranges
color_dict_HSV = {'black': [[180, 255, 44], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'pink' : [[169, 255, 255], [160, 45, 45]],
              'red1': [[180, 255, 255], [170, 45, 45]],
              'red2': [[9, 255, 255], [0, 45, 45]],
              'green': [[86, 255, 255], [36, 45, 45]],
              'blue': [[128, 255, 255], [87, 45, 45]],
              'yellow': [[35, 255, 255], [16, 45, 45]],
              'purple': [[159, 255, 255], [129, 45, 45]],
              'orange': [[15, 255, 255], [10, 45, 45]],
              'gray': [[180, 18, 230], [0, 0, 101]],
              'darkgray': [[180, 18, 100], [0, 0, 45]]}

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


def all_colors_segment_bbox(roi):
    ksize = (3,3)
    blur = cv2.blur(roi, ksize, cv2.BORDER_DEFAULT) 

    # run code
    all_holds = []
    all_colors = []
    all_contours = []
    for color in ['red', 'blue', 'green', 'purple', 'yellow', 'white', 'pink', 'black', 'orange']:
        mask = segment_color(color, blur)
        holds, contours = find_bounds(mask)
        all_contours += contours
        all_holds += holds 
        all_colors += [color]*len(holds)

    all_holds, all_colors, all_contours = filter_bounds(all_holds, all_colors, all_contours)

#     draw_contours(all_contours, all_colors, roi)

    # 	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    draw_bounds(all_holds, all_colors, roi)

    # returns list of holds and corresponding list with the color of each hold
    return all_holds, all_colors, all_contours

    


def segment_color(color, rgb_img):
	"""
	Segments a color out of the image
	"""
	# convert to hsv colorspace
	hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

	# lower bound and upper bound for Yellow color
	#lower_bound = np.array([20, 80, 80])	 
	#upper_bound = np.array([30, 255, 255])
	if color == 'red':
		lower_bound = np.array(color_dict_HSV['red1'][1])
		upper_bound = np.array(color_dict_HSV['red1'][0])

		# find the colors within the boundaries
		mask1 = cv2.inRange(hsv, lower_bound, upper_bound)

		lower_bound = np.array(color_dict_HSV['red2'][1])
		upper_bound = np.array(color_dict_HSV['red2'][0])

		# find the colors within the boundaries
		mask2 = cv2.inRange(hsv, lower_bound, upper_bound)
		mask = cv2.bitwise_or(mask1, mask2) 

	else:
		lower_bound = np.array(color_dict_HSV[color][1])
		upper_bound = np.array(color_dict_HSV[color][0])

		# find the colors within the boundaries
		mask = cv2.inRange(hsv, lower_bound, upper_bound)

	#define kernel size  
	kernel = np.ones((5,5),np.uint8)

	# Remove unnecessary noise from mask
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	# Segment only the detected region
	#segmented_img = cv2.bitwise_and(img, img, mask=mask)

	return mask

def find_bounds(mask,w_thresh=200,h_thresh=200):
	"""
	Finds the bounding boxes given a mask
	"""
	# Find contours from the mask
	contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# get bounding boxes
	holds = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if w > w_thresh or h > h_thresh:
			continue
		holds.append([(x,y),(x+w,y+h)])

	return holds, contours

def draw_contours(contours, colors, rgb_img):
	"""
	Draws the contours of an image
	"""
	img_cp = np.ones((rgb_img.shape[0],rgb_img.shape[1],rgb_img.shape[2]))*255
	output = cv2.drawContours(img_cp, contours, -1, (0,0,0), 3)
	for i, cnt in enumerate(contours):
		output = cv2.drawContours(img_cp, cnt, -1, color_dict_rgb[colors[i]], 3)
	plt.imshow(img_cp.astype(int))
	plt.show()
	return img_cp

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

def overlap(hold1, hold2):
	"""
	Determines if hold bounding boxes overlap
	"""
	x1_low, y1_low = hold1[0]
	x1_hi, y1_hi = hold1[1]
	x2_low, y2_low = hold2[0]
	x2_hi, y2_hi = hold2[1]

	r1 = (x1_hi >= x2_low and x2_hi >= x1_low)
	r2 = (y1_hi >= y2_low and y2_hi >= y1_low)
	overlap = r1 and r2

	return overlap

def filter_bounds(all_holds, all_colors, all_contours):
	"""
	Filters bounding boxes to make output more percise
	Removes overlapping boxes
	"""
	rm_idx = []

	for i in range(len(all_holds)):
		for j in range(i+1, len(all_holds)):
			if all_colors[i]==all_colors[j]:
				continue
			hold1 = all_holds[i]
			hold2 = all_holds[j]
			if overlap(hold1,hold2): # decide if a box should be removed
				x1_low, y1_low = hold1[0]
				x1_hi, y1_hi = hold1[1]
				x2_low, y2_low = hold2[0]
				x2_hi, y2_hi = hold2[1]

				# remove smaller box?
				a1 = (x1_hi - x1_low) * (y1_hi - y1_low)
				a2 = (x2_hi - x2_low) * (y2_hi - y2_low)

				if a1 > a2:
					rm_idx.append(j)
				if a2 > a1:
					rm_idx.append(i)

	new_holds = []
	new_colors = []
	new_contours = []
	for i in range(len(all_holds)):
		if i in rm_idx:
			continue
		else:
			new_holds.append(all_holds[i])
			new_colors.append(all_colors[i])
			new_contours.append(all_contours[i])
	return new_holds, new_colors, new_contours


def all_colors_segment(rgb_img, mask):
	"""
    Takes in an RGB Image of a hold wall, and a wall mask and segments image based on colors

    returns: list(list(tuples)) each sublist is [(x_min, y_min), (x_max, y_max)]
                and represents a single hold
             list(strings) where each string is the name of a color
    """

	# # Reading the image
	# img = cv2.imread(fn)
	# # Convert to rgb
	# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	#define kernel size  
	kernel = np.ones((23,23),np.uint8)
	# Remove unnecessary noise from mask
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	# mask sum increases during this noise removal?

	rgb_img = cv2.bitwise_and(mask, rgb_img)
	# plt.imshow(rgb_img)
	# plt.show()

	# Show original image
	#plt.imshow(rgb_img)
	#plt.show()

	# blur image
	ksize = (9,9)
	blur = cv2.blur(rgb_img, ksize, cv2.BORDER_DEFAULT) 

	# run code
	all_holds = []
	all_colors = []
	all_contours = []
	for color in ['red', 'blue', 'green', 'purple', 'yellow', 'white', 'pink', 'black', 'orange']:
		mask = segment_color(color, blur)
		holds, contours = find_bounds(mask)
		all_contours += contours
		all_holds += holds 
		all_colors += [color]*len(holds)

	all_holds, all_colors, all_contours = filter_bounds(all_holds, all_colors, all_contours)

	# draw_contours(all_contours, all_colors, rgb_img)

	# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# draw_bounds(all_holds, all_colors, rgb_img)

	# returns list of holds and corresponding list with the color of each hold
	return all_holds, all_colors, all_contours

