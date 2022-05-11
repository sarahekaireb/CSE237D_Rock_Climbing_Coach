# -*- coding: utf-8 -*-
"""Pose_Estimation.ipynb

Automatically generated by Colaboratory.
"""

# !pip install mediapipe

# from google.colab import drive
# drive.mount('/content/drive')

import cv2
import mediapipe as mp
import time
from scipy import spatial
import math
from math import hypot


# initialize mediapipe requirements
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# input video path
cap = cv2.VideoCapture('rock_dataset_0/clip2/climb.mp4')

# global dict to store the coordiantes (required towards MVP)
dict_coordinates = {'left_hand': [], 'right_hand': [], 'left_leg': [], 'right_leg': [], 'left_hip': [], 'right_hip': []}

# compute landmarks for a frame
def find_pose(img):
  break_signal = False
  results = []
  try:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print('Landmarks:', results.pose_landmarks)
  except:
    break_signal = True

  return img, results, break_signal

# retrieve coordinates from lm_list and store the coordinates in the global dict
def store_coordinates(lm_list):
  global dict_coordinates
  dict_coordinates['left_hand'].append((lm_list[38], lm_list[39])) #left_index - x, y 
  dict_coordinates['right_hand'].append((lm_list[40], lm_list[41])) #right_index - x, y
  dict_coordinates['left_hip'].append((lm_list[46], lm_list[47])) #left_hip - x, y
  dict_coordinates['right_hip'].append((lm_list[48], lm_list[49])) #right_hip - x, y
  dict_coordinates['left_leg'].append((lm_list[62], lm_list[63])) #left_foot - x, y
  dict_coordinates['right_leg'].append((lm_list[64], lm_list[65])) #right_foot - x, y

# compute cosine smilarity between two lm lists 
def check_similarity(list1, list2):
  result = 1 - spatial.distance.cosine(list1, list2)
  return result

# plot the image only when the frames are dissimilar
def plot_image(img, results, cx, cy, pTime):
  mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
  cv2.circle(img, (cx, cy), 5, (255,0, 150), cv2.FILLED)
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
    
  cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
  cv2.imshow('image', img)
  cv2.waitKey(1)

def compute_distance(list1, list2):
  list_1_left_x = list1[46]
  list_1_left_y = list1[47]
  list_2_left_x = list2[46]
  list_2_left_y = list2[46]
  left_dist = math.hypot(list_1_left_x-list_2_left_x, list_1_left_y-list_2_left_y)
  list_1_right_x = list1[48]
  list_1_right_y = list1[49]
  list_2_right_x = list2[48]
  list_2_right_y = list2[49]
  right_dist = math.hypot(list_1_right_x-list_2_right_x, list_1_right_y-list_2_right_y)
  return (left_dist + right_dist)/2

def check_coord_bounding(lm_list, holds_bounding_box):
  if check_point_in_box(lm_list[38], lm_list[39], holds_bounding_box) and check_point_in_box(lm_list[40], lm_list[41], holds_bounding_box) and check_point_in_box(lm_list[62], lm_list[63], holds_bounding_box) and check_point_in_box(lm_list[64], lm_list[65], holds_bounding_box):
      return True

def check_point_in_box(x,y, holds_bounding_box):
  return True

def main(cap):
  prev = []
  output_img_list = []
  first_frame_flag = True
  total_frames_count = 0
  stored_frames_count = 0
  pTime = 0
  total_distance = 0
  num_moves = 0
  holds_bounding_box = []

  while True:
    print('----------------------')
    print('Processing a new frame')
    success, img = cap.read()
    img, results, main_break_signal = find_pose(img)
    
    # the signal means that there are no more input frames in the video, and thus the code must terminate
    if (main_break_signal == True):
      break
    
    lm_list = []

    if results.pose_landmarks:     
      # add all 66 cordinates to lm_list
      for id, lm in enumerate(results.pose_landmarks.landmark):  
        h, w, c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        lm_list.append(cx)
        lm_list.append(cy)

      # for the first frame, compute and store the coordinates
      if(first_frame_flag == True):
        store_coordinates(lm_list)
        print("Similarity found for the first frame and coordinates stored")
        prev = lm_list
        first_frame_flag = False
        stored_frames_count += 1
        output_img_list.append(img)
        plot_image(img, results, cx, cy, pTime)


      # from next frame onwards, first check similarity and then store the coordinates
      else:
        result = check_similarity(prev, lm_list) #prev = 66 cordinates, lm_list = 66 cordinates
        print('Similarity Value:', result)
        if(result < 0.9999):
          store_coordinates(lm_list)
          if check_coord_bounding(lm_list, holds_bounding_box):
            total_distance += compute_distance(prev, lm_list)
            num_moves = num_moves + 1
          print("Similarity found and coordinates stored")
          stored_frames_count += 1
          output_img_list.append(img)
          plot_image(img, results, cx, cy, pTime)
        
        prev = lm_list
          
      print('Prev list: ', prev)
      print('Length of prev list: ', len(prev))
      print('LM list: ', lm_list)
      print('Length of lm_list: ', len(lm_list))

    total_frames_count += 1

  print('---------- Processsing Completed ----------')
  print('Total frames processed: ', total_frames_count)
  print('Total frames stored: ', stored_frames_count)
  print('Total distance covered (in pixels): ', total_distance)
  print('Number of moves: ', num_moves)
  
  # output a video consisting of just the processed frames
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float 'width'
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float 'height'
  img_size = (width, height)
  fps = 5
  output_video = cv2.VideoWriter('pose_estimation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, img_size)
 
  for i in range(len(output_img_list)):
    output_video.write(output_img_list[i])
  
  output_video.release()
  print('Output Video Released: ', fps, 'fps')

def get_last_frame_cordinates(cap):
  last_frame_cordinates = []

  #Run code on input video - cap and store coordinates of all frames in dict_coordinates
  main(cap)

  #Return last frame coordinates
  last_frame_cordinates.append(dict_coordinates['left_hand'][-1])
  last_frame_cordinates.append(dict_coordinates['right_hand'][-1])
  last_frame_cordinates.append(dict_coordinates['left_hip'][-1])
  last_frame_cordinates.append(dict_coordinates['right_hip'][-1])
  last_frame_cordinates.append(dict_coordinates['left_leg'][-1])
  last_frame_cordinates.append(dict_coordinates['right_leg'][-1])
  
  return (last_frame_cordinates)

if __name__ == "__main__":
    main(cap)