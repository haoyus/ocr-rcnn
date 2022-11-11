import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

# this file is only for using Torchvision version of Faster RCNN model

def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == '__main__':
    model = create_model(10)
    model.eval()
    image = torch.randn((1,3,480,640), dtype=torch.float32)
    # with torch.no_grad():
    #     outputs = model(image)
    #     print(outputs)
    torch.onnx.export(
        model,
        image,
        "faster_rcnn.onnx",
        opset_version=11,
        input_names = ['input'],
        output_names = ['boxes', 'labels', 'scores']
    )