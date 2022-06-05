import io
import cv2
import torch
from torchvision import transforms
import numpy as np
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import argparse
import utils.color_range_analysis_utils as cvpred
from utils.color_range_analysis_utils import all_colors_segment
from utils.train_utils.model_factory import get_segmentation_model
from utils.color_holds_prediction_utils import getAllHoldColors

# remove from script when publishing to GitHub
MODEL = 'hold-detection'
VERSION = '1'
API_KEY = '3cZO2UYZLwtFu4j2STv0' 


def correctHolds(img,wall):
    kernel = np.ones((23,23),np.uint8)
    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask[mask<127] = 0
    mask[mask>=127] = 255
#     img = cv2.bitwise_and(img,mask)
#     _,holds =predict_holds(img)
#     holds = process_hold_response(holds)
    holds,mlcolors = predict_NN_holds_colors(img)
#     _,mlcolors = mlpred.getAllHoldColors(img,holds)
    # colors = ["black"]*len(holds)
#     colors2 = ["red"]*len(color_holds)
    newHolds = []
    newColors = []
    # img = mlpred.draw_bounds(holds,colors2,img)
    # img = mlpred.draw_bounds(color_holds,colors,img)
    for j in range(len(holds)):
        hold = holds[j]
#         print("ml color ",mlcolors[j])
        xmin,ymin = hold[0]
        xmax,ymax = hold[1]
        # print("(%d,%d),(%d,%d)"%(xmin,xmax,ymin,ymax))
        roi = img[ymin:ymax,xmin:xmax]
        cholds,ccolors,contours = cvpred.all_colors_segment_bbox(roi)
#         mask = cvpred.segment_color(mlcolors[j],roi)
#         holds1, contours = cvpred.find_bounds(mask)
#         cvpred.draw_bounds(holds1,[mlcolors[j]]*len(holds1),roi)
        maxArea = 0
        idx = -1
        ci = None
        # print(len(cholds))
        if(len(cholds)==0):
            newHolds.append(hold)
            newColors.append("black")
            continue
        for i in range(len(cholds)):
            h = cholds[i]
            x1,y1 = h[0]
            x2,y2 = h[1]
            W = x2-x1
            H = y2-y1
            a = W*H
            # print(a)
            if(a>maxArea):
                maxArea = a
                idx = i
                ci = ccolors[i]
        x1,y1 = cholds[idx][0]
        x2,y2 = cholds[idx][1]
        if(np.sum(mask[ymin+y1,xmin+x1]) == 0 and np.sum(mask[ymin+y2,xmin+x2])==0):
#             print("*******************SKIPPING*********************")
            continue
#         print("Max area ",maxArea,"index",idx)
        percentage = 100*maxArea/((xmax-xmin)*(ymax-ymin))
#         print("Percentage covered ",100*maxArea/((xmax-xmin)*(ymax-ymin)))
        if(percentage>55):
            newHolds.append([(xmin+x1,ymin+y1),(xmin+x2,ymin+y2)])
        else:
            newHolds.append(hold)
            
        # print("(%d,%d),(%d,%d)"%(xmin,xmax,ymin,ymax))
    #     plt.imshow(roi)
        
        newColors.append(ci)
    return newHolds, newColors



# def correctHolds_2(img):
# #     _,holds =predict_holds(img)
# #     holds = process_hold_response(holds)
#     holds, mlcolors = predict_NN_holds_colors(img)
# #     _,mlcolors = mlcolors
#     print("holds",holds)
#     print("mlcolors",mlcolors)
# #     _,mlcolors = mlpred.getAllHoldColors(img,holds)
#     # colors = ["black"]*len(holds)
# #     colors2 = ["red"]*len(color_holds)
#     newHolds = []
#     newColors = []
#     # img = mlpred.draw_bounds(holds,colors2,img)
#     # img = mlpred.draw_bounds(color_holds,colors,img)
#     for j in range(len(holds)):
#         hold = holds[j]
#         print("ml color ",mlcolors[j])
#         xmin,ymin = hold[0]
#         xmax,ymax = hold[1]
#         # print("(%d,%d),(%d,%d)"%(xmin,xmax,ymin,ymax))
#         roi = img[ymin:ymax,xmin:xmax]
# #         cholds,ccolors,contours = cvpred.all_colors_segment_bbox(roi)
#         mask = cvpred.segment_color(mlcolors[j],roi)
#         cholds, contours = cvpred.find_bounds(mask)
# #         cvpred.draw_bounds(holds1,[mlcolors[j]]*len(holds1),roi)
#         maxArea = 0
#         idx = -1
#         ci = None
#         print(len(cholds))
#         if(len(cholds)==0):
#             newHolds.append(hold)
#             newColors.append("black")
#             continue
#         for i in range(len(cholds)):
#             h = cholds[i]
#             x1,y1 = h[0]
#             x2,y2 = h[1]
#             W = x2-x1
#             H = y2-y1
#             a = W*H
#             print(a)
#             if(a>maxArea):
#                 maxArea = a
#                 idx = i
# #                 ci = ccolors[i]
#         x1,y1 = cholds[idx][0]
#         x2,y2 = cholds[idx][1]
#         print("Max area ",maxArea,"index",idx)
#         percentage = 100*maxArea/((xmax-xmin)*(ymax-ymin))
#         print("Percentage covered ",100*maxArea/((xmax-xmin)*(ymax-ymin)))
#         if(percentage>52):
#             newHolds.append([(xmin+x1,ymin+y1),(xmin+x2,ymin+y2)])
#         else:
#             newHolds.append(hold)
            
#         # print("(%d,%d),(%d,%d)"%(xmin,xmax,ymin,ymax))
#     #     plt.imshow(roi)
        
#         newColors.append(ci)
#     return newHolds


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

def get_wall_mask(rgb_img, wall_model=None, wall_model_loc='../models/wall_segmentor.pth'):
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
    holds, colors, contours = all_colors_segment(rgb_img, wall_mask, isMask = True)
    return holds, colors, wall_model

def predict_holds_colors(rgb_img, wall_model=None):
    wall_mask, wall_model = get_wall_mask(rgb_img, wall_model=wall_model) # keep wall model to prevent needing to reload model
    holds, colors = correctHolds(rgb_img, wall_mask)
    return holds, colors, wall_model