import mediapipe as mp
from scipy import spatial

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