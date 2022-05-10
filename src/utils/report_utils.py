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

