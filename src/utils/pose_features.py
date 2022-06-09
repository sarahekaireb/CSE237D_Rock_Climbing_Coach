from utils.pc_complete_utils import joint_in_hold
import math
from matplotlib import pyplot as plt

def get_num_moves(climb_holds_used, frame_significances):
    assert len(climb_holds_used) == len(frame_significances)

    move_holds_used = []
    prev_holds_used = None
    for i in range(len(frame_significances)):
        if frame_significances[i] and prev_holds_used is None:
            move_holds_used.append(climb_holds_used[i])
            prev_holds_used = climb_holds_used[i]
        elif frame_significances[i]:
            if sum(climb_holds_used[i]) > 0 and climb_holds_used[i] != prev_holds_used:
                move_holds_used.append(climb_holds_used[i])
                prev_holds_used = climb_holds_used[i]

    distinct_holds_used = list(set([tuple(elem) for elem in move_holds_used]))

    num_moves = len(move_holds_used) - 1
    
    return num_moves, move_holds_used, distinct_holds_used

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

def compute_total_distance_traveled(dir, sig_positions):
    prev_left = sig_positions['left_hip'][0]
    prev_right = sig_positions['right_hip'][0]
    
    total_dist = 0
    distances = []
    for i in range(1, len(sig_positions['left_hip'])):
        cur_left = sig_positions['left_hip'][i]
        cur_right = sig_positions['right_hip'][i]
        
        delta = compute_distance(prev_left, prev_right, cur_left, cur_right)
        total_dist += delta
        distances.append(delta)
        prev_left = cur_left
        prev_right = cur_right
    
    generate_distance_graph(dir, distances)
    return total_dist

def generate_distance_graph(dir, distances):
    plt.plot(distances)
    plt.xlabel('nth move', labelpad=15)
    plt.ylabel('Distance', labelpad=15)
    plt.title('Distance vs nth move')
    plt.savefig(dir+'/distance_moved.png')