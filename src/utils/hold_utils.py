import io
import cv2
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import argparse

# remove from script when publishing to GitHub
MODEL = 'hold-detection'
VERSION = '1'
API_KEY = '3cZO2UYZLwtFu4j2STv0' 

def predict_holds(rgb_img):
    """
    Returns response status for request to roboflow model API
        as well as JSON output of hold predictions

    return: tuple(status code, json obj)
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
    return response, response.json()

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