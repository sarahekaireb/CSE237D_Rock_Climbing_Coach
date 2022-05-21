import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# HOLD DETECTION MODEL IS HOSTED ON ROBOFLOW API
# ONLY WALL SEGMENTATION MODEL IS TRAINED BY US

# def get_object_detection_model(model_name='FasterRCNN'):
#     if model_name == 'FasterRCNN':
#         # load a model pre-trained pre-trained on COCO
#         model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#         # get number of input features for the classifier
#         in_features = model.roi_heads.box_predictor.cls_score.in_features
#         # replace the pre-trained head with a new one
#         model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) 

#     return model

def get_segmentation_model(model_name='FCN_Resnet50'):
    if model_name == 'FCN_Resnet50':
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[-1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[-1] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

        # initialize output conv2d
        nn.init.kaiming_normal_(model.classifier[-1].weight)
        nn.init.kaiming_normal_(model.classifier[-1].weight)

    return model
