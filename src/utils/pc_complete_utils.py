import numpy as np

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

def get_last_double_handhold(holds, positions):
    last_idx = -1
    zipped = list(zip(positions['left_hand'], positions['right_hand']))
    for i in range(len(zipped)):
        lh, rh = zipped[i]
        for j in range(len(holds)):
            if joint_in_hold(lh, holds[j]) and joint_in_hold(rh, holds[j]):
                last_idx = j
    return last_idx

def get_lowest_hold_used(holds, positions):
    lowest_y = 0
    lowest = -1
    
    zipped = list(zip(positions['left_hand'], positions['right_hand'], positions['left_leg'], positions['right_leg']))
    for i in range(len(zipped)):
        lh, rh, ll, rl = zipped[i]
        for j in range(len(holds)):
            if joint_in_hold(lh, holds[j]) or joint_in_hold(rh, holds[j]) or joint_in_hold(ll, holds[j]) or joint_in_hold(rl, holds[j]):
                hold_min, hold_max = holds[j]
                # hold_x = int(hold_max[0] - ((hold_max[0] - hold_min[0]) / 2))
                hold_y = int(hold_max[1] - ((hold_max[1] - hold_min[1]) / 2))
                if hold_y > lowest_y: # pixels start (0, 0) at top left
                    lowest_y = hold_y
                    lowest = j
    return lowest, lowest_y

def get_highest_hold_unused(holds):
    highest_y = np.inf
    highest = -1
    for i in range(len(holds)):
        hold_min, hold_max = holds[i]
        hold_y = int(hold_max[1] - ((hold_max[1] - hold_min[1]) / 2))
        
        if hold_y < highest_y: # pixels start (0, 0) at top left
            highest_y = hold_y
            highest = i
    return highest, highest_y

def compute_percent_complete(holds, positions):
    """Returns % of Route Completed

    holds: list(list(tuple)) bbox coordinates of detected holds
    positions: dic keys are joints, values are lists of coordinates at all timesteps
    
    returns: float indicating % of Route completed
    """
    last_handhold_idx = get_last_double_handhold(holds, positions)
    if last_handhold_idx == -1:
        raise AssertionError("Both hands were never on the same hold")
    else:
        last_min, last_max = holds[last_handhold_idx]
        # last_x = int(last_max[0] - ((last_max[0] - last_min[0]) / 2))
        last_y = int(last_max[1] - ((last_max[1] - last_min[1]) / 2))

        lowest, lowest_y = get_lowest_hold_used(holds, positions)
        highest, highest_y = get_highest_hold_unused(holds)

        climbed = last_y - lowest_y
        could_climb = highest_y - lowest_y

        print("Highest: ", highest_y, highest)
        print("Lowest: ", lowest_y, lowest)
        print("Last: ", last_y, last_handhold_idx)

        return (climbed / could_climb) * 100

#############################
## Using color information ##
#############################

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


def get_last_double_handhold_color(holds, positions, colors, color):
    last_idx = -1
    zipped = list(zip(positions['left_hand'], positions['right_hand']))
    for i in range(len(zipped)):
        lh, rh = zipped[i]
        for j in range(len(holds)):
            if joint_in_hold(lh, holds[j]) and joint_in_hold(rh, holds[j]) and colors[j] == color:
                last_idx = j
    return last_idx

def get_lowest_hold_used_color(holds, positions, colors, route_color):
    lowest_y = 0
    lowest = -1
    
    zipped = list(zip(positions['left_hand'], positions['right_hand'], positions['left_leg'], positions['right_leg']))
    for i in range(len(zipped)):
        lh, rh, ll, rl = zipped[i]
        for j in range(len(holds)):
            if joint_in_hold(lh, holds[j]) or joint_in_hold(rh, holds[j]) or joint_in_hold(ll, holds[j]) or joint_in_hold(rl, holds[j]):
                if colors[j] == route_color:
                    hold_min, hold_max = holds[j]
                    # hold_x = int(hold_max[0] - ((hold_max[0] - hold_min[0]) / 2))
                    hold_y = int(hold_max[1] - ((hold_max[1] - hold_min[1]) / 2))
                    if hold_y > lowest_y: # pixels start (0, 0) at top left
                        lowest_y = hold_y
                        lowest = j
    return lowest, lowest_y

def get_highest_hold_unused_color(holds, colors, route_color):
    highest_y = np.inf
    highest = -1
    for i in range(len(holds)):
        hold_min, hold_max = holds[i]
        hold_y = int(hold_max[1] - ((hold_max[1] - hold_min[1]) / 2))
        
        if hold_y < highest_y and colors[i] == route_color: # pixels start (0, 0) at top left
            highest_y = hold_y
            highest = i
    
    return highest, highest_y


def get_last_handhold(holds, positions, colors, route_color):
    # gets the lower hand position in the frame which had the highest hand position
    hand_positions = list(zip(positions['left_hand'], positions['right_hand']))
    highest_hand_pos = np.inf
    highest_hand_frame = -1
    for i in range(len(hand_positions)):
        left_hand, right_hand = hand_positions[i]
    
        for h in range(len(holds)):
            if (joint_in_hold(left_hand, holds[h]) or joint_in_hold(right_hand, holds[h])) and colors[h] == route_color:
                mins, maxs = holds[h]
                cy = (maxs[1] - mins[1]) / 2 + mins[1] #y_min + y_max - y_min
                if cy < highest_hand_pos:
                    highest_hand_pos = cy
                    highest_hand_frame = i
    
    left_highest, right_highest = hand_positions[highest_hand_frame]
    highest_lower_hand_pos = min(left_highest[1], right_highest[1])
    return highest_lower_hand_pos

def compute_percent_complete_color(holds, colors, positions):
    print("Color PC Complete")
    holds_used = get_holds_used(holds, positions)
    colors_used = []
    for t in range(len(holds_used)): # outer loop over frames
        for i in range(len(holds_used[t])):
            if holds_used[t][i]:
                colors_used.append(colors[i])
    
    # route color is most frequent hold color
    route_color = max(set(colors_used), key=colors_used.count)

    lowest, lowest_y = get_lowest_hold_used_color(holds, positions, colors, route_color)
    highest, highest_y = get_highest_hold_unused_color(holds, colors, route_color)
    last_y = get_last_handhold(holds, positions, colors, route_color)
    climbed = last_y - lowest_y
    could_climb = highest_y - lowest_y

    # print("Highest: ", highest_y, highest)
    # print("Lowest: ", lowest_y, lowest)
    # print("Last: ", last_y, last_handhold_idx)

    return (climbed / could_climb) * 100


    # last_handhold_idx = get_last_double_handhold_color(holds, positions, colors, route_color)
    # if last_handhold_idx == -1:
    #     last_handhold_idx = get_last_double_handhold(holds, positions)
    #     raise AssertionError("Both hands were never on the same color hold")
    # else:
    #     last_min, last_max = holds[last_handhold_idx]
    #     last_y = int(last_max[1] - ((last_max[1] - last_min[1]) / 2))

    #     lowest, lowest_y = get_lowest_hold_used_color(holds, positions, colors, route_color)
    #     highest, highest_y = get_highest_hold_unused_color(holds, colors, route_color)

    #     climbed = last_y - lowest_y
    #     could_climb = highest_y - lowest_y

    #     print("Highest: ", highest_y, highest)
    #     print("Lowest: ", lowest_y, lowest)
    #     print("Last: ", last_y, last_handhold_idx)

    #     return (climbed / could_climb) * 100



