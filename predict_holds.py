import io
import cv2
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import argparse

MODEL = 'hold-detection'
VERSION = '1'
API_KEY = '3cZO2UYZLwtFu4j2STv0'

def get_parser():
    parser = argparse.ArgumentParser('Command line utility for hold detection')
    parser.add_argument('-i', '--image_path', type=str, help='path to image for hold detection')
    return parser

def predict_holds(rgb_img):
    """
    Returns response status for request to roboflow model API
        as well as JSON output of hold predictions

    return: tuple(status code, json obj)
    """
    # Load Image with PIL
    # img = cv2.imread("/Users/wolf/Downloads/P7.jpg")
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    print(response)
    print(response.json())

    return response, response.json()

if __name__ == '__main__':
    # test
    parser = get_parser()
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict_holds(rgb_image)
    print("done")