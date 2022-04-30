import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from model_factory import get_segmentation_model

import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser('Command line utility for hold detection')
    parser.add_argument('-i', '--image_path', type=str, help='path to image for hold detection')
    parser.add_argument('-m', '--model_path', default='./models/wall_segmentor.pth', type=str,
                        help='path to segmentation model on local disk')
    return parser

def preprocess(img_tensor):
    """
    Preprocesses an image so that it can be fed into the wall-segmentation model

    img_tensor: tensor of shape 3 x H x W
    """
    img_min = img_tensor.flatten(start_dim=1).min(dim=1).values.view(-1, 1, 1)
    img_max = img_tensor.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1)

    minmax_normed = (img_tensor - img_min) / (img_max - img_min)
    normalized = F.normalize(minmax_normed, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    return normalized

def predict_wall(rgb_image, model_path):
    """
    rgb_image: np.array H x W x 3
    """
    model = get_segmentation_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()

    img = torch.LongTensor(rgb_image).permute(2, 0, 1) # make channels first dim
    img = preprocess(img) # normalize
    img = img.unsqueeze(0) # make batch

    preds = model(img)['out'] # batch of predictions
    mask = preds[0].argmax(dim=0) # binary mask of wall segmentation for img (1 == wall)
    
    return mask.cpu().numpy()

if __name__ == '__main__':
    # test script
    parser = get_parser()
    args = parser.parse_args()

    rgb_image = np.array(Image.open(args.image_path))
    mask = predict_wall(rgb_image, model_path=args.model_path)
    print('done!')

