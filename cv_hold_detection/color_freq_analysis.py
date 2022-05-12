#! /usr/bin/python

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

def plot_colors_peaks(fn, peak_hues, hold_window=4, v_thresh=30, s_thresh=30):
	"""
	Given a list of peak hues, generate a plot for each hue
	"""
	img = cv2.imread(fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	for peak_hue in peak_hues:
		peak_img = np.zeros(hsv_img.shape, np.uint8)
		for i in range(hsv_img.shape[0]):
			for j in range(hsv_img.shape[1]):
				h,s,v = hsv_img[i,j]
				if peak_hue-hold_window <= h and h <= peak_hue+hold_window:
					if v>v_thresh and s>s_thresh:
						peak_img[i,j] = hsv_img[i,j]
		rgb_img = cv2.cvtColor(peak_img, cv2.COLOR_HSV2RGB)
		plt.imshow(rgb_img.astype(int))
		myRGB = rgb_img.astype(int)
		plt.show()

def find_peaks_histogram(hist, window=5, thresh=200):
	"""
	find the peaks in the color histogram
	"""
	flag = 1
	peak_hues = []
	while flag == 1:
		# get max hue value
		cur_hue = np.argmax(hist)
		peak_hues.append(cur_hue)

		# left
		i = cur_hue-1
		while(hist[i] > hist[i-1]-15):
			hist[i] = 0
			i = i-1

		# right
		i = cur_hue+1
		while(hist[i] > hist[i+1]+15):
			hist[i] = 0
			i = i+1

		for i in range(cur_hue-window,cur_hue+window+1):
			if i < len(hist) and i >= 0:
				hist[i] = 0
		if np.max(hist) < thresh:
			flag = 0
	return peak_hues

def hue_histogram(fn, v_thresh=50, s_thresh=50):
	"""
	returns a histogram of the hue values in the image for v > thresh, s > thresh
	"""
	img = cv2.imread(fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	hist = np.zeros([180])
	for i in range(hsv_img.shape[0]):
		for j in range(img.shape[1]):
			h,s,v = hsv_img[i,j]
			if v > v_thresh and s > s_thresh:
				hist[h] += 1
	return hist

def generate_color_plot(fn):
	"""
	Generate HSV color plot of image pixels
	"""
	img = cv2.imread(fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
	norm = colors.Normalize(vmin=-1.,vmax=1.)
	norm.autoscale(pixel_colors)
	pixel_colors = norm(pixel_colors).tolist()

	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	h, s, v = cv2.split(hsv_img)
	fig = plt.figure()
	axis = fig.add_subplot(1, 1, 1, projection="3d")

	axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
	axis.set_xlabel("Hue")
	axis.set_ylabel("Saturation")
	axis.set_zlabel("Value")
	plt.show()


if __name__ == "__main__":
	if len(sys.argv)!=2: # Expect exactly one arguments: filename of hold wall image
		sys.exit(2)

	fn = sys.argv[1]
	generate_color_plot(fn)
	cp_a = hue_histogram(fn)
	peak_hues = find_peaks_histogram(cp_a)
	plot_colors_peaks(fn, peak_hues)
	