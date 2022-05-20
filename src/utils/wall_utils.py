import cv2
import numpy as np
from scipy import stats

def estimateBackground(vid_fn, sz=25):
	"""
	Estimates the background of the image by giving the median image
	"""

	# open video 
	vid = cv2.VideoCapture(vid_fn)

	# randomly select subset of frames
	frameIds = vid.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=sz)
	#frameIds = vid.get(cv2.CAP_PROP_FRAME_COUNT) * np.linspace(0, 0.9999, sz)

	frames = []
	for fid in frameIds:
		vid.set(cv2.CAP_PROP_POS_FRAMES, fid)
		ret, frame = vid.read()
		frames.append(frame)

	# Calculate the median along the time axis
	medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)  

	# convert to rgb
	rgb_background = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2RGB)

	# return the cropped image
	return cropImg(rgb_background)


def cropImg(rgb_img):
	"""
	Crops the black out of the sides of an image
	"""

	# convert to gray
	gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

	# binary threshold image
	_,thresh = cv2.threshold(gray_img,1,255,cv2.THRESH_BINARY)

	#define kernel size  
	kernel = np.ones((7,7),np.uint8)
	# Remove unnecessary noise from thresh
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	# find contours
	contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)

	# crop image
	crop = rgb_img[y:y+h,x:x+w]

	return crop
