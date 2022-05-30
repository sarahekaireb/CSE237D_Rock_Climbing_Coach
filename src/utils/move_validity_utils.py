#! /usr/bin/python

import cv2
import video_utils
import pose_utils
import hold_utils
import color_range_analysis_utils
import numpy as np
import mediapipe as mp

def getFirstHoldColor(holds_used, holds, colors):
	"""
	Gets color of first hold used
	"""
	for i in range(len(holds_used)):
		if True in holds_used[i]:
			color_route = getMostFrequentHoldColor([holds_used[i]], holds, colors)
			break
	return color_route

def getMostFrequentHoldColor(holds_used, holds, colors):
	"""
	Gets the most frequent color hold used
	""" 

	all_holds_used = np.logical_or.reduce(holds_used)

	hold_colors_used = {}
	for i in range(len(all_holds_used)):
		if all_holds_used[i]:
			if colors[i] in hold_colors_used:
				hold_colors_used[colors[i]] += 1
			else:
				hold_colors_used[colors[i]] = 1
			

	color_route = max(hold_colors_used, key=hold_colors_used.get)
	print("route of color: " + color_route)

	return color_route

def getMostFrequentHoldColorHovered(holds_used, holds, colors):
	"""
	Gets the most frequent color hold used
	""" 

	hold_colors_used = []
	for i in range(len(holds_used)):
		for j in range(len(holds_used[i])):
			if holds_used[i][j]:
				hold_colors_used.append(colors[j])

	color_route = max(set(hold_colors_used), key=hold_colors_used.count)
	print("route of color: " + color_route)

	return color_route

def getColorRoute(holds_used, holds, colors, mode = 'hovered'):
	"""
	Gets the color of the route the climber is using
	"""
	if mode == 'freq':
		color_route = getMostFrequentHoldColor(holds_used, holds, colors)
	elif mode == 'first':
		color_route = getFirstHoldColor(holds_used, holds, colors)
	elif mode == 'hovered':
		color_route = getMostFrequentHoldColorHovered(holds_used, holds, colors)

	return color_route


def create_video(vid_arr, holds, colors, holds_used, color_route, pose_results, dict_coordinates, frame_significances):
	"""
	Creates the move validity video
	"""
	joint_list = list(zip(dict_coordinates['left_hand'], 
						  dict_coordinates['right_hand'], 
						  dict_coordinates['left_hip'], 
						  dict_coordinates['right_hip'], 
						  dict_coordinates['left_leg'], 
						  dict_coordinates['right_leg']))
	mp.solutions.drawing_utils.draw_landmarks
	plotted_frames = []
	
	for t in range(vid_arr.shape[0]):
		if frame_significances[t] == False:
			continue
		else:
			print(t)
			used = holds_used[t]
			results = pose_results[t]

			frame = vid_arr[t]
			print("Drawing Holds")
			for h in range(len(used)):
				# drawing holds used
				if used[h]:
					if colors[h] == color_route:
						frame = cv2.rectangle(frame, holds[h][0], holds[h][1], (0, 255, 120), 5)
					else:
						frame = cv2.rectangle(frame, holds[h][0], holds[h][1], (255, 50, 50), 5)
			# draw pose
			print("Drawing Pose")
			mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

			# draw keypoints
			print("Drawing Keypoints")
			for j in range(len(joint_list[t])):
				cx, cy = joint_list[t][j]
				frame = cv2.circle(frame, (cx, cy), 5, (255,0, 150), cv2.FILLED)
			plotted_frames.append(frame)
	return np.array(plotted_frames)

def get_holds_used(holds, dict_coordinates):
	# should return a list of lists
	# nested list should be an array of True/False
	# each index of the nested list will correspond 
	# to the hold at that index in holds
	
	joint_list = list(zip(dict_coordinates['left_hand'], dict_coordinates['right_hand'], dict_coordinates['left_leg'], dict_coordinates['right_leg']))
	holds_used = []
	for i in range(len(joint_list)): # frames
		used_arr = []
		for h in range(len(holds)):
			hold = holds[h]
			joint_usage = [joint_in_hold(joint, hold) for joint in joint_list[i]]
			if sum(joint_usage) >= 1:
				try: # checking if next frame also uses same hold
					next_joint_usage = [joint_in_hold(joint, hold) for joint in joint_list[i+1]]
					if sum(next_joint_usage) >= 1:
						used_arr.append(True)
					else:
						used_arr.append(False)
				except:
					used_arr.append(True)
			else:
				used_arr.append(False)
		holds_used.append(used_arr)
	return holds_used

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


def process_video(VIDEO_PATH, HOLDS_PATH, hd_mode = 'cv'):
	# get the video and significant frames
	video = video_utils.get_video_array(VIDEO_PATH)

	frames, results_arr, landmarks_arr, joint_positions = pose_utils.get_video_pose(video)

	significances = pose_utils.get_significant_frames(landmarks_arr)

	# get the holds and colors
	# using ml
	if hd_mode == 'ml':
		print("ml")
		hold_img = video[0]
		#hold_img = cv2.cvtColor(cv2.imread(HOLDS_PATH), cv2.COLOR_BGR2RGB)
		holds, colors = hold_utils.predict_NN_holds_colors(hold_img)
	else:
		print("cv")
		hold_img = video[0]
		#hold_img = cv2.cvtColor(cv2.imread(HOLDS_PATH), cv2.COLOR_BGR2RGB)
		holds, colors, contours = color_range_analysis_utils.all_colors_segment(hold_img)

	climb_holds_used = get_holds_used(holds, joint_positions)


	return video, climb_holds_used, holds, colors, frames, results_arr, landmarks_arr, joint_positions, significances


def runMoveValidity(VIDEO_PATH, HOLDS_PATH, hd_mode = 'cv'):
	video, climb_holds_used, holds, colors, frames, results_arr, landmarks_arr, joint_positions, significances = process_video(VIDEO_PATH, HOLDS_PATH, hd_mode)

	color_route = getColorRoute(climb_holds_used, holds, colors, mode = 'hovered')

	video = video.take(frames, axis=0)

	plotted_vid = create_video(video, holds, colors, climb_holds_used, color_route, results_arr, joint_positions, significances)

	# write video to file
	fourcc= cv2.VideoWriter_fourcc(*'MJPG')
	height,width,layers=plotted_vid[0].shape
	out = cv2.VideoWriter('move_validity.avi', fourcc, 5, (width, height), isColor=True)

	for i in range(plotted_vid.shape[0]):
		out.write(cv2.cvtColor(plotted_vid[i], cv2.COLOR_RGB2BGR))

	out.release()

	return plotted_vid











