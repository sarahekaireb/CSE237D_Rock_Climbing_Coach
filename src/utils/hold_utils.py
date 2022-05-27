import io
import cv2
import torch
from torchvision import transforms
import numpy as np
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import argparse

from .color_range_analysis_utils import all_colors_segment
from .train_utils.model_factory import get_segmentation_model
from .color_holds_prediction_utils import getAllHoldColors

# remove from script when publishing to GitHub
MODEL = 'hold-detection'
VERSION = '1'
API_KEY = '3cZO2UYZLwtFu4j2STv0' 

def process_hold_response(dic):
    """
    Processes the json response from hold detection API

    returns: list(list(tuples)) each sublist is [(x_min, y_min), (x_max, y_max)]
                and represents a single hold
    """
    # can union with color frequency analysis in this method
    hold_arr = dic['predictions']
    holds = []
    for elem in hold_arr:
        center_x, center_y = elem['x'], elem['y']
        width, height = elem['width'], elem['height']
        
        x_min = int(center_x - width/2)
        y_min = int(center_y - height/2)
        
        x_max = int(center_x + width/2)
        y_max = int(center_y + height/2)
        
        hold = [(x_min, y_min), (x_max, y_max)]
        holds.append(hold)
    return holds

def predict_NN_holds_colors(rgb_img):
    """
    Requests hold predictions from Model API and returns predicted holds
    along with predicted colors of each hold

    Colors are derived using the center coordinate of each hold bounding box

    returns: list(list(tuple)) holds, list(string) colors
    """
    pilImage = Image.fromarray(rgb_img)

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")

    # Build multipart form and post request
    m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
    url = "https://detect.roboflow.com/{model}/{version}?api_key={api}".format(
                                                        model=MODEL, 
                                                        version=VERSION, 
                                                        api=API_KEY)
    response = requests.post(url, data=m, headers={'Content-Type': m.content_type})
    try:
        holds = process_hold_response(response.json())
        colors = getAllHoldColors(rgb_img, holds)
        print("Colors: ", len(colors))
        return holds, colors
    except:
        raise Exception("API Failed to return successful response or no holds detected")

def get_wall_mask(rgb_img, wall_model=None, wall_model_loc='../../models/wall_segmentor.pth'):
    """
    Helper method to predict wall-segmentation mask
    for use in CV hold-color detection
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if wall_model is None:
        print("Loading Wall-Segmentation Model")
        wall_model = get_segmentation_model().to(device)
        wall_model.load_state_dict(torch.load(wall_model_loc, map_location=device))
        wall_model.eval()
        print("Model Loaded.")

    # convert img to normalized tensor for prediction of wall segmentation mask
    img = torch.LongTensor(rgb_img).permute(2, 0, 1) # 3 x H x W
    img_min = img.flatten(start_dim=1).min(dim=1).values.view(-1, 1, 1)
    img_max = img.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1)
    minmax_normed = (img - img_min) / (img_max - img_min)
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_norm = norm(minmax_normed)

    batch = img_norm.unsqueeze(0).to(device) # 1 x 3 x H x W
    with torch.no_grad():
        dic = wall_model(batch)
        preds = dic['out'] # N x C x H x W
        preds = torch.argmax(preds, dim=1, keepdim=False) # N x H x W
        mask = preds[0].unsqueeze(2).repeat(1, 1, 3)
        mask = (mask.cpu().numpy() * 255).astype(np.uint8) # H x W
    
    return mask, wall_model

def predict_CV_holds_colors(rgb_img, wall_model=None):
    """
    Returns holds and associated colors using Color-Frequency Analysis Method
    of Hold detection

    returns: list(list(tuple)) holds, list(string) colors, FCN-Resnet-50 Segmentation Model
    """
    # predict wall mask
    # use color freq to get holds, etc
    # return holds
    wall_mask, wall_model = get_wall_mask(rgb_img, wall_model=wall_model) # keep wall model to prevent needing to reload model
    holds, colors, contours = all_colors_segment(rgb_img, wall_mask)
    return holds, colors, wall_model