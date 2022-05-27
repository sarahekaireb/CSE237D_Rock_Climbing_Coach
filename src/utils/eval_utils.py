"""
This file is intended to contain logic and methods
used for evaluating functionalities needed for
the report generation.
"""
import os
import numpy as np
import cv2
import json

from video_utils import *
from pose_utils import *
from hold_utils import predict_CV_holds_colors, predict_NN_holds_removeWall, predict_NN_holds_colors
from pc_complete_utils import compute_percent_complete

import gc

class HoldEvaluator:
    def __init__(self, method, dataset_loc='../../datasets/uncropped_holds_coco'):
        self.dataset_loc = dataset_loc
        self.modes = ['train', 'test', 'valid']
        assert method in ['NN', 'color_freq', 'NN_wall'] # accepted methods of Hold Detection
        self.method = method
        self.wall_model = None
        self.load_data()

    def load_data(self):
        """loads dataset of hold images for evaluation"""
        self.data = {}
        for mode in self.modes:
            print("Loading {} Data".format(mode.upper()))
            mode_dir = os.path.join(self.dataset_loc, mode)
            annot_fp = os.path.join(mode_dir, '_annotations.coco.json')
            with open(annot_fp, 'r') as f:
                mode_annots = json.load(f)
            
            mode_images = {}
            for fn in os.listdir(mode_dir):
                if fn.endswith('.jpg'):
                    img_path = os.path.join(mode_dir, fn)
                    img = cv2.imread(img_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    mode_images[fn] = rgb_img

            mode_mapping = {}
            for elem in mode_annots['images']:
                img_id = elem['id']
                img_fn = elem['file_name']
                mode_mapping[img_fn] = img_id

            self.data[mode] = {'annots': mode_annots['annotations'], 'images': mode_images, 'mapping': mode_mapping}

    def annot_to_holds(self, img_annots):
        # img_annots list of dics, bbox in pixel coordinates
        holds = []
        for elem in img_annots:
            x_min, y_min, w, h = elem['bbox']
            x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
            x_max = x_min + w
            y_max = y_min + h

            hold = [(x_min, y_min), (x_max, y_max)]
            holds.append(hold)
        return holds

    def holds_to_mask(self, img, holds):
        # returns binary mask of holds
        mask = np.zeros((img.shape[0], img.shape[1])) # H x W
        for elem in holds:
            tup1, tup2 = elem
            x_min, y_min = tup1
            x_max, y_max = tup2

            mask[y_min:y_max+1, x_min:x_max+1] = 1
        return mask

    def dice_score(self, pred_mask, true_mask):
        # https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
        pred_mask = pred_mask.flatten()
        true_mask = true_mask.flatten()

        true_inds = (true_mask == 1)
        TP = pred_mask[true_inds].sum() # all correct + predictions

        TP_FP = pred_mask.sum() # TP + FP (ie. all + preds)
        TP_FN = true_mask.sum() # TP + FN (ie. all + ground truths)

        return (2*TP) / (TP_FP + TP_FN)

    def calculate_dice(self, img, img_annots):
        dice = 0
        if self.method == 'NN':
            pred_holds, colors = predict_NN_holds_colors(img)
        elif self.method == 'color_freq':
            pred_holds, colors, wall_model = predict_CV_holds_colors(img, wall_model=self.wall_model)
            self.wall_model = wall_model
        elif self.method == 'NN_wall':
            pred_holds, colors, self.wall_model = predict_NN_holds_removeWall(img, wall_model=self.wall_model)
        # _, response = predict_holds(img)
        # pred_holds = process_hold_response(response) # list of [(x_min, y_min), (y_min, y_max)]
        true_holds = self.annot_to_holds(img_annots) # list of [(x_min, y_min), (y_min, y_max)]

        pred_mask = self.holds_to_mask(img, pred_holds) # H x W
        true_mask = self.holds_to_mask(img, true_holds) # H x W

        dice = self.dice_score(pred_mask, true_mask)
        return dice

    def evaluate(self):
        mean_dice = 0
        num_images = 0
        
        for mode in self.modes:
            print("Calculating DICE for {} mode".format(mode.upper()))
            annots = self.data[mode]['annots']
            images = self.data[mode]['images'] # dictionary of fn-nparray
            mapping = self.data[mode]['mapping'] # dictionary of fn to img_id
            for fn in mapping:
                img_id = mapping[fn]
                img = images[fn]
                img_annots = [elem for elem in annots if elem['image_id'] == img_id]
                img_dice = self.calculate_dice(img, img_annots) # prediction happens in this method
                num_images += 1
                mean_dice += img_dice 
        
        mean_dice = mean_dice / num_images
        return mean_dice

if __name__ == '__main__':
    # Hold Evaluation
    h_NN_eval = HoldEvaluator(method='NN')
    NN_avg_dice = h_NN_eval.evaluate()

    h_CV_eval = HoldEvaluator(method='color_freq')
    CV_avg_dice = h_CV_eval.evaluate()

    h_NN_wall_eval = HoldEvaluator(method='NN_wall')
    NN_wall_avg_dice = h_NN_wall_eval.evaluate()

    print("\nAverage NN DICE: ", NN_avg_dice)
    print("Average CV DICE: ", CV_avg_dice)
    print("Average NN Wall Remove DICE: ", NN_wall_avg_dice)