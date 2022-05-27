import cv2
import mediapipe as mp
import time
from scipy import spatial
from IPython.core.display import ProgressBar
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

import argparse

from utils.video_utils import *
from utils.pose_utils import *
from utils.hold_utils import *
from utils.pc_complete_utils import compute_percent_complete, compute_percent_complete_color

# constants for cropping video
NEW_WIDTH = 576
NEW_HEIGHT = 1080

def get_parser():
    parser = argparse.ArgumentParser(description='Run script to produce a report from a climb video')
    parser.add_argument('-d', '--dir', type=str, default='../test_data',
                        help='filepath of climb video and hold image for generating report.txt')
    return parser

def main(args):
    files = os.listdir(args.dir)
    assert 'climb.mp4' in files
    assert 'holds.jpg' in files
    vid_path = os.path.join(args.dir, 'climb.mp4')
    img_path = os.path.join(args.dir, 'holds.jpg')

    raw_vid = get_video_array(vid_path)
    hold_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # All pose-related information
    frames, results_arr, landmarks_arr, joint_positions = get_video_pose(raw_vid)
    raw_vid = raw_vid.take(frames, axis=0)
    significances = get_significant_frames(landmarks_arr)

    climb_holds, hold_colors = predict_NN_holds_colors(hold_img)
    try:
        assert len(climb_holds) == len(hold_colors)
    except:
        print("{} Holds".format(len(climb_holds)))
        print("{} Colors".format(len(hold_colors)))
        print(hold_colors)
        print(climb_holds)
        raise AssertionError("Each hold does not have an associated color")
    # percent_complete = compute_percent_complete(climb_holds, joint_positions)
    percent_complete = compute_percent_complete_color(climb_holds, hold_colors, joint_positions)
    print("Completed ", str(percent_complete) + '% of climb')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)