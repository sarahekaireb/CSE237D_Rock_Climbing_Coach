import mediapipe as mp
from scipy import spatial
import numpy as np
import math
from scipy.signal import savgol_filter
import scipy
from matplotlib import pyplot as plt

def get_video_pose(vid_arr):
    """
    Returns all pose information from a video
    vid_arr: np array

    returns:
        all_results: list of mediapipe results
        all_landmarks: list(list) each sublist contains all landmarks for a specific frame
        dict_coordinates: keys are joints, values are list(tuple) 
                            each tuple is x,y coordinate for that joint at a specific frame
    """
    pose = mp.solutions.pose.Pose()
    
    dict_coordinates = {'left_hand': [], 'right_hand': [], 'left_hip': [], 'right_hip': [], 'left_leg': [], 'right_leg': []}
    all_landmarks = []
    all_results = []
    frames = []
    for i in range(vid_arr.shape[0]):
        img = vid_arr[i]
        results = pose.process(img)
        
        if results.pose_landmarks is not None:
            frames.append(i)
            lm_list = []
            for id, lm in enumerate(results.pose_landmarks.landmark):  
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append(cx)
                lm_list.append(cy)
            all_landmarks.append(lm_list)
            all_results.append(results)
            dict_coordinates['left_hand'].append((lm_list[38], lm_list[39])) #left_index - x, y 
            dict_coordinates['right_hand'].append((lm_list[40], lm_list[41])) #right_index - x, y
            dict_coordinates['left_hip'].append((lm_list[46], lm_list[47])) #left_hip - x, y
            dict_coordinates['right_hip'].append((lm_list[48], lm_list[49])) #right_hip - x, y
            dict_coordinates['left_leg'].append((lm_list[62], lm_list[63])) #left_foot - x, y
            dict_coordinates['right_leg'].append((lm_list[64], lm_list[65])) #right_foot - x, y
    
    return frames, all_results, all_landmarks, dict_coordinates

def check_similarity(list1, list2):
    """Returns similarity between two lists of landmark coordinates"""
    result = 1 - spatial.distance.cosine(list1, list2)
    return result

def find_troughs(angles):
    angles = np.asarray(angles)
    peaks = scipy.signal.find_peaks(-1*angles,distance = 21,width=5)
    return peaks

def get_significant_frames_motion_graph(dir,landmarks):
    L = len(landmarks)
    angles = []
    step = 7
    for i in range(step,L):
        lm_list1 = np.asarray(landmarks[i-step])
        lm_list2 = np.asarray(landmarks[i])
        idx = [38,39,40,41,46,47,48,49,62,63,64,65]
        
        pose_1 = lm_list1[idx]
        pose_2 = lm_list2[idx]
        angle = math.acos(1-spatial.distance.cosine(list(pose_1),list(pose_2)))
        angles.append(angle)
    angles_smooth = savgol_filter(angles,21,3)
    peaks = find_troughs(angles_smooth)[0]
    sig_frames = np.zeros(L)
    sig_frames[peaks] = 1
    generate_motion_graph(dir, angles)
    return sig_frames

def generate_motion_graph(dir, angles):
    plt.plot(angles)
    plt.ylabel('Angle', labelpad=15)
    plt.title('Angle vs nth move')
    plt.savefig(dir+'/angular_velocity.png')

def get_significant_frames(landmarks):
    """
    Uses similarity to identify significant frames within a video
    landmarks: list(list) each sublist has all landmark coordinates from pose information
    
    returns: list(boolean) True indicates frame at this position is important, False otherwise
    """
    significant = []
    for i in range(len(landmarks)):
        if i == 0:
            significant.append(True)
        else:
            if check_similarity(prev, landmarks[i]) < 0.99999:
                significant.append(True)
            elif i == len(landmarks)-1:
                significant.append(True)
            else:
                significant.append(False)
        prev = landmarks[i]
    return significant