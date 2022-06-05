from utils.pc_complete_utils import joint_in_hold
import math

def get_num_moves(climb_holds_used, frame_significances):
    assert len(climb_holds_used) == len(frame_significances)

    # num_moves = 0
    # prev_holds_used = all_holds_used[0]
    # for i in range(1, len(frame_significances)):
    #     if frame_significances[i]:
    #         if sum(all_holds_used[i]) > 0 and all_holds_used[i] != prev_holds_used: # 30/26
    #             num_moves += 1
    #             prev_holds_used = all_holds_used[i]

    # for i in range(1, len(all_holds_used)): # 50
    #     if all_holds_used[i] != all_holds_used[i-1]:
    #         num_moves += 1
    
    # num_moves = len(set([tuple(elem) for elem in all_holds_used])) - 1 # 16
    
    sig_holds_used = []
    for i in range(1, len(frame_significances)):
        if frame_significances[i]:
            sig_holds_used.append(tuple(climb_holds_used[i]))
    distinct_holds_used = set(sig_holds_used)
    num_moves = len(distinct_holds_used)
    
    return num_moves, list(distinct_holds_used)

def compute_time_elapsed(video, holds, all_positions, fps=30):
    # start frame is first frame where both hands are on some hold
    start_frame = -1
    for i in range(video.shape[0]):
        for hold in holds:
            if joint_in_hold(all_positions['left_hand'][i], hold) and joint_in_hold(all_positions['right_hand'][i], hold):
                start_frame = i
                break
        if start_frame != -1:
            break
    
    end_frame = -1
    highest_hip = video.shape[1]
    for i in range(video.shape[0]):
        left_hip = all_positions['left_hip'][i]
        right_hip = all_positions['right_hip'][i]
        
        hip_y_pos = min(left_hip[1], right_hip[1])
        if hip_y_pos < highest_hip:
            highest_hip = hip_y_pos
            end_frame = i
    
    if start_frame == -1 or end_frame == -1:
        return -1
    else:
        return (end_frame - start_frame + 1) / fps # time elapsed in seconds

def compute_distance(left_hip1, right_hip1, left_hip2, right_hip2):
    l1_x, l1_y = left_hip1
    l2_x, l2_y = left_hip2

    r1_x, r1_y = right_hip1
    r2_x, r2_y = right_hip2

    left_dist = math.hypot(l1_x - l2_x, l1_y - l2_y)
    right_dist = math.hypot(r1_x - r2_x, r1_y - r2_y)
    return (left_dist + right_dist)/2

def compute_total_distance_traveled(sig_positions):
    prev_left = sig_positions['left_hip'][0]
    prev_right = sig_positions['right_hip'][0]
    
    total_dist = 0
    for i in range(1, len(sig_positions['left_hip'])):
        cur_left = sig_positions['left_hip'][i]
        cur_right = sig_positions['right_hip'][i]
        
        delta = compute_distance(prev_left, prev_right, cur_left, cur_right)
        total_dist += delta

        prev_left = cur_left
        prev_right = cur_right
    
    return total_dist