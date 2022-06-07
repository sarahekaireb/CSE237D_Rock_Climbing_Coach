import mediapipe as mp
from scipy import spatial

from utils.hold_utils import get_holds_and_colors
from utils.pc_complete_utils import get_holds_used
from utils.pose_utils import get_significant_frames_motion_graph

def check_similarity(list1, list2):
    """Returns similarity between two lists of landmark coordinates"""
    result = 1 - spatial.distance.cosine(list1, list2)
    return result

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
    significances = []
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
            
#             if i == 0:
#                 significances.append(True)
#             elif check_similarity(prev, lm_list) < 0.99999:
#                 significances.append(True)
#             else:
#                 significances.append(False)
            
            prev = lm_list
        
            all_landmarks.append(lm_list)
            all_results.append(results)
            dict_coordinates['left_hand'].append((lm_list[38], lm_list[39])) #left_index - x, y 
            dict_coordinates['right_hand'].append((lm_list[40], lm_list[41])) #right_index - x, y
            dict_coordinates['left_hip'].append((lm_list[46], lm_list[47])) #left_hip - x, y
            dict_coordinates['right_hip'].append((lm_list[48], lm_list[49])) #right_hip - x, y
            dict_coordinates['left_leg'].append((lm_list[62], lm_list[63])) #left_foot - x, y
            dict_coordinates['right_leg'].append((lm_list[64], lm_list[65])) #right_foot - x, y
    significances = get_significant_frames_motion_graph(all_landmarks)

    return frames, all_results, all_landmarks, dict_coordinates, significances

def process_video(video, hold_img):
    # joint_positions stores all landmarks for all frames -- even insignificant frames
    # significances denotes whether the frame was significant
    frames, results_arr, landmarks_arr, all_positions, significances = get_video_pose(video)
    video = video.take(frames, axis=0)

    sig_positions = {'left_hand': [], 'right_hand': [], 'left_hip': [], 'right_hip': [], 'left_leg': [], 'right_leg': []}
    for i in range(len(significances)):
        if significances[i]:
            sig_positions['left_hand'].append(all_positions['left_hand'][i])
            sig_positions['right_hand'].append(all_positions['right_hand'][i])
            sig_positions['left_hip'].append(all_positions['left_hip'][i])
            sig_positions['right_hip'].append(all_positions['right_hip'][i])
            sig_positions['left_leg'].append(all_positions['left_leg'][i])
            sig_positions['right_leg'].append(all_positions['right_leg'][i])

    # num moves should be sum(significances) - 1

    # holds, colors, wall_model = predict_holds_colors(hold_img, wall_model=None)    
    holds, colors = get_holds_and_colors(hold_img)
    climb_holds_used = get_holds_used(holds, all_positions)

    return video, climb_holds_used, holds, colors, results_arr, landmarks_arr, all_positions, sig_positions, significances
