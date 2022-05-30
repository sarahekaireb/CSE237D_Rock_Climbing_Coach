# -*- coding: utf-8 -*-
"""Pose_Estimation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18INMq8rpmFKCu5vQfK4NgPMPnqW0OJus
"""

#!pip install mediapipe
#!pip install requests_toolbelt

#from google.colab import drive
#drive.mount('/content/drive')

import cv2
import mediapipe as mp
import time
from scipy import spatial
import math
from math import hypot
#from google.colab.patches import cv2_imshow # if using colab
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from predict_holds import predict_holds
from matplotlib import pyplot as plt
from src.report import retrieve_objects

# initialize mediapipe requirements
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# input video path
cap = cv2.VideoCapture('/content/drive/MyDrive/CSE_237D/rock_dataset2/clip12/cropped.mp4')
img = cv2.imread('/content/drive/MyDrive/CSE_237D/rock_dataset2/clip12/holds.jpg')
cap_holds = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input ground truth excel file name (corresponding to Cosine method and Total time elapsed)
excel_path = '/content/drive/MyDrive/CSE_237D/climb_seconds_test.xlsx'
excel_path_total_elapsed_time = '/content/drive/MyDrive/CSE_237D/time_elapsed_test.xlsx'

# input clip name, format: Clip 2 -> the format has to correspond with the sheet name in both excel file above
clip_name = 'Clip 12'

# input directory path including both image and video -> to be used for total time elapsed
dir_path = '/content/drive/MyDrive/CSE_237D/rock_dataset2/clip12'


# define the cosine similarity threshold 
threshold = 0.9999

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
def plot_image(img, results, cx, cy, elapsed_time):
  mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
  cv2.circle(img, (cx, cy), 5, (255,0, 150), cv2.FILLED)
  cv2.putText(img, str(int(elapsed_time)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
  #cv2_imshow(img) # if using colab
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

def check_coord_bounding(lm_list, holds_bounding_box, h, w):
  if (check_point_in_box(lm_list[62], lm_list[63], holds_bounding_box, h , w) 
  or check_point_in_box(lm_list[64], lm_list[65], holds_bounding_box, h , w)
  or check_point_in_box(lm_list[30], lm_list[31], holds_bounding_box, h , w) 
  or check_point_in_box(lm_list[32], lm_list[33], holds_bounding_box, h , w)):
      print("found valid position!!")
      return True
  return False

def check_point_in_box(cx,cy, holds_bounding_box, h, w):
  for box in holds_bounding_box:
    if(cx > box[0] and cx < box[2] and cy > box[1] and cy < box[3]):
      #print(box, " ", cx, " ", cy)
      return True

  return False

# load ground truth labels from excel 
def load_from_excel(excel_path):
  excel_pd = pd.ExcelFile(excel_path)
  raw_ground_truth_dict = {sheet : excel_pd.parse(sheet)['Timings'].tolist() for sheet in excel_pd.sheet_names}
  # print(raw_ground_truth_dict['Clip 2'])
  total_time_ground_truth_dict = {sheet : excel_pd.parse(sheet)['Timings'].tolist() for sheet in excel_pd.sheet_names}
  return raw_ground_truth_dict, total_time_ground_truth_dict

# compute accuracy of predictions
def compute_scores(raw_predictions, raw_ground_truth_labels, total_elapsed_time):

  # masked attributes are those which can be directly used with sklearn metric functions to compute the scores
  masked_predictions = [0 for time in range(total_elapsed_time + 1)]
  masked_ground_truth_labels = [0 for time in range(total_elapsed_time + 1)]

  # perform masking; don't forget to add 1 to total_elapsed_time when using range()
  for time in range(total_elapsed_time + 1):
    if time in raw_predictions:
      masked_predictions[time] = 1
    
    if time in raw_ground_truth_labels:
      masked_ground_truth_labels[time] = 1

  print('masked_predictions: ', masked_predictions)
  print('masked_ground_truth_labels: ', masked_ground_truth_labels)

  # the format for using sklearn scores is: score(y_true, y_pred)
  accuracy = accuracy_score(masked_ground_truth_labels, masked_predictions)
  precision = precision_score(masked_ground_truth_labels, masked_predictions)
  recall = recall_score(masked_ground_truth_labels, masked_predictions)
  f1score = f1_score(masked_ground_truth_labels, masked_predictions)

  return accuracy, precision, recall, f1score

def parse_holds(holds):
  pred_list = holds[1]['predictions']
  bounding_boxes = []

  for p in pred_list:
    min_x = p['x']-p['width']/2
    min_y = p['y']-p['height']/2
    max_x = p['x']+p['width']/2
    max_y = p['y']+p['height']/2

    box = [min_x, min_y, max_x,max_y]
    bounding_boxes.append(box)

  return bounding_boxes

def joint_in_hold(joint, hold):
    # joint is (x, y)
    # hold is [(x_min, y_min), (x_max, y_max)]
    jx, jy = joint
    h_xmin, h_ymin = hold[0]
    h_xmax, h_ymax = hold[1]
    
    if jx <= h_xmax and jx >= h_xmin and jy <= h_ymax and jy >= h_ymin:
        return True
    else:
        return False

def compute_time_elapsed(fps, total_elapsed_time_ground_truth):
  # retrieve the objects from report.py
  raw_vid, climb_holds, joint_positions = retrieve_objects(dir_path)
  
  # compute the first frame such that both hands are on some hold
  start_frame = -1
  start_flag = False
  # prepare a zipped list from left_hand and right_hand positions
  hands_zipped = list(zip(joint_positions['left_hand'], joint_positions['right_hand']))
  # loop through each frame in raw_vid
  for frame_index in range(raw_vid.shape[0]):
    # compute left hand and right hand positions
    left_hand, right_hand = hands_zipped[frame_index]
    # loop through the holds
    for hold_index in range(len(climb_holds)):
      if (joint_in_hold(left_hand, climb_holds[hold_index]) and joint_in_hold(right_hand, climb_holds[hold_index])):
        start_frame = frame_index
        start_flag = True
        break
    if(start_flag):
      break

  # compute the last frame, i.e., the frame in which the positions of hips is the highest
  end_frame = -1
  # (0,0) starts at top left, hence the least y (closest to 0 y coordinate) will represent max height
  min_hip_y_pos = 99999
  # prepare a zipped list from left_hand and right_hand positions
  hips_zipped = list(zip(joint_positions['left_hip'], joint_positions['right_hip']))
  # loop through each set of hip coordinate
  for frame_index in range(len(hips_zipped)):
    left_hip, right_hip = hips_zipped[frame_index]
    # taking mid point gives wrong results, hence take the max of left_hip, right_hip
    hip_y_pos = min(left_hip[1], right_hip[1])
    if hip_y_pos <= min_hip_y_pos:
      min_hip_y_pos = hip_y_pos
      end_frame = frame_index

  # check for validity
  if (start_frame == -1 or end_frame == -1):
    print('Couldn\'t compute the total elapsed time: check the logic')
    return -1, -1, total_elapsed_time_ground_truth, -1
  
  else:
    # compute the total time elapsed
    total_time_elapsed_predicted = (end_frame - start_frame + 1)/fps;
    # compute the accuracy of prediction
    error_rate = abs(total_elapsed_time_ground_truth - total_time_elapsed_predicted)/ total_elapsed_time_ground_truth
    accuracy = 1 - error_rate

    return start_frame, end_frame, total_time_elapsed_predicted, accuracy

def main(cap, cap_holds):
  prev = []
  output_img_list = []
  first_frame_flag = True
  stored_frames_count = 0
  pTime = 0
  total_distance = 0
  distances = []
  num_moves = 0
  holds = predict_holds(cap_holds)
  print("holds")
  holds_bounding_box = parse_holds(holds)
  print(holds_bounding_box)
  # required for computing accuracy
  raw_predictions = []

  # required for time elapased and accuracy computation
  fps = cap.get(cv2.CAP_PROP_FPS) # note that this fps is constant (we shot the videos at 30 fps)
  
  # compute the total time elapsed -> required for masking the raw predictions and raw ground truth labels
  
  # total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  # load the ground truth from excel
  _, total_time_ground_truth_dict = load_from_excel(excel_path_total_elapsed_time) # each key of the total_time_ground_truth_dict represents a clip and the corresponding value is the list of timesteps
  
  # retrieve ground truth labels (from total_time_ground_truth_dict) only for the current clip 
  start_time_ground_truth = total_time_ground_truth_dict[clip_name][0]
  start_frame_ground_truth = fps * start_time_ground_truth
  end_time_ground_truth = total_time_ground_truth_dict[clip_name][1]
  end_frame_ground_truth = fps * end_time_ground_truth

  # compute total time (ground truth) by subtracting start time from end time
  total_elapsed_time_ground_truth = end_time_ground_truth - start_time_ground_truth

  # compute total frame count (we output this towards the final results)
  total_frame_count = fps * total_elapsed_time_ground_truth

  # compute predictions for total time elapsed
  start_frame_predicted, end_frame_predicted, total_time_elapsed_predicted, accuracy_time_elapsed = compute_time_elapsed(fps, total_elapsed_time_ground_truth)
  print('----------------------')
  print('Start Frame (ground truth): ', start_frame_ground_truth)
  print('End frame (ground truth): ', end_frame_ground_truth)
  print('Start Frame (predicted): ', start_frame_predicted)
  print('End frame (predicted): ', end_frame_predicted)

  while True:
    print('----------------------')
    print('Processing a new frame')
    success, img = cap.read()
    img, results, main_break_signal = find_pose(img)
      
    # the signal means that there are no more input frames in the video, and thus the code must terminate
    if (main_break_signal == True):
      break
    
    # compute the frame number for the current frame
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print('Current Frame Number: ', frame_number)

    # process only those frames which fall within the total time elapsed (i.e., exclude frames preceding and succeeding the actual climb) 
    if (frame_number >= start_frame_predicted and frame_number <= end_frame_predicted):
      # compute the elapsed time for current frame
      elapsed_time = frame_number / fps
      # cTime = time.time()
      # fps = 1 / (cTime - pTime)
      # pTime = cTime
      print('Elapsed time: ', elapsed_time)
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
          plot_image(img, results, cx, cy, elapsed_time)
          raw_predictions.append(int(elapsed_time)) # conversion to int just cuts the decimal part, which is actually the intended behavior followed by ground truth videos


        # from next frame onwards, first check similarity and then store the coordinates
        else:
          result = check_similarity(prev, lm_list) #prev = 66 cordinates, lm_list = 66 cordinates
          print('Similarity Value:', result)
          if(result < threshold):
            store_coordinates(lm_list)
            if check_coord_bounding(lm_list, holds_bounding_box, h, w):
              curr_dist = compute_distance(prev, lm_list)
              total_distance += curr_dist
              num_moves = num_moves + 1
              distances.append(curr_dist)
            print("Similarity found and coordinates stored")
            stored_frames_count += 1
            output_img_list.append(img)
            plot_image(img, results, cx, cy, elapsed_time)
            raw_predictions.append(int(elapsed_time)) # conversion to int just cuts the decimal part, which is actually the intended behavior followed by ground truth videos
          
          prev = lm_list
            
        print('Prev list: ', prev)
        print('Length of prev list: ', len(prev))
        print('LM list: ', lm_list)
        print('Length of lm_list: ', len(lm_list))

    # the frame was preceding/succeeding the actual climb
    else:
      print('Current frame precedes/succeeds the actual climb - coordinates not stored')

  print('---------- Processsing Completed ----------')
  # using total_time_elapsed_predicted instead of ground truth
  print('Total elapsed time (ground truth): ', total_elapsed_time_ground_truth)
  print('Total elapsed time (predicted): ', total_time_elapsed_predicted)
  print('Accuracy for total elapsed time : ', accuracy_time_elapsed)
  print('Total frames processed: ', total_frame_count)
  print('Total frames stored: ', stored_frames_count)
  print('Total distance covered (in pixels): ', total_distance)
  print('Number of moves: ', num_moves)
  
  # output a video consisting of just the processed frames
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float 'width'
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float 'height'
  img_size = (width, height)
  fps_output = 5
  output_video = cv2.VideoWriter('pose_estimation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps_output, img_size)
 
  for i in range(len(output_img_list)):
    output_video.write(output_img_list[i])
  
  output_video.release()
  print('Output Video Released: ', fps_output, 'fps')

  # computing accuracy
  # load the ground truth from excel
  raw_ground_truth_dict, _ = load_from_excel(excel_path) # each key of the raw_ground_truth_dict represents a clip and the corresponding value is the list of timesteps
  
  # retieve ground truth labels (from raw_ground_truth_dict) only for the current clip 
  raw_ground_truth_labels = raw_ground_truth_dict[clip_name]

  # retrieve the evaluation metric scores, using total_time_elapsed_predicted instead of ground truth
  accuracy, precision, recall, f1_score = compute_scores(raw_predictions, raw_ground_truth_labels, int(total_time_elapsed_predicted))
  print('Cosine similarity threshold used: ', threshold)
  print('Accuracy: ', accuracy)
  print('Precision: ', precision)
  print('Recall: ', recall)
  print('F-1 Score: ', f1_score)
  print('Distance per jump:' , distances)
  plt.plot(distances)
  plt.xlabel('nth move', labelpad=15)
  plt.ylabel('Distance', labelpad=15)
  plt.title('Distance vs nth move')
  plt.show()
  plt.savefig('distance_moved.png')

if __name__ == "__main__":
    main(cap, cap_holds)

def get_last_frame_cordinates(cap):
  last_frame_cordinates = []

  #Run code on input video - cap and store coordinates of all frames in dict_coordinates
  main(cap, cap_holds)

  #Return last frame coordinates
  last_frame_cordinates.append(dict_coordinates['left_hand'][-1])
  last_frame_cordinates.append(dict_coordinates['right_hand'][-1])
  last_frame_cordinates.append(dict_coordinates['left_hip'][-1])
  last_frame_cordinates.append(dict_coordinates['right_hip'][-1])
  last_frame_cordinates.append(dict_coordinates['left_leg'][-1])
  last_frame_cordinates.append(dict_coordinates['right_leg'][-1])
  
  return (last_frame_cordinates)