import cv2
import numpy as np
from PIL import Image

def get_video_array(vid_path):
    """Returns np array of video, given a path"""
    cap = cv2.VideoCapture(vid_path)
    to_read = True
    vid_arr = []
    while to_read:
        to_read, frame = cap.read()
        if to_read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_arr.append(frame)
    return np.array(vid_arr)

def crop_video(vid_arr, new_width, new_height):
    """
    Crops a video to a desired resolution
    vid_arr: np array
    new_width: int
    new_height: int
    """
    cropped_video = []
    for i in range(vid_arr.shape[0]):
        frame = vid_arr[i]
        im = Image.fromarray(frame)
        
        width, height = im.size   # Get dimensions

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        cropped = im.crop((left, top, right, bottom))
        cropped_video.append(np.array(cropped))
    return np.array(cropped_video)