import torch
import math

def bboxes_to_grasps(bbox):
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
    # box = torch.unbind(bboxes, axis=1)
    x = (bbox[0] + (bbox[4] - bbox[0])/2) * 0.35
    y = (bbox[1] + (bbox[5] - bbox[1])/2) * 0.47
    tan = (bbox[3] -bbox[1]) / (bbox[2] - bbox[0]) *0.47/0.35
    h = torch.sqrt(torch.pow((bbox[2] -bbox[0])*0.35, 2) + torch.pow((bbox[3] - bbox[1])*0.47, 2))
    w = torch.sqrt(torch.pow((bbox[6] -bbox[0])*0.35, 2) + torch.pow((bbox[7] - bbox[1])*0.47, 2))
    return x, y, tan, h, w

def grasp_to_bbox(x, y, tan, h, w):
    theta = torch.atan(tan)
    edge1 = (x -w/2*torch.cos(theta) +h/2*torch.sin(theta), y -w/2*torch.sin(theta) -h/2*torch.cos(theta))
    edge2 = (x +w/2*torch.cos(theta) +h/2*torch.sin(theta), y +w/2*torch.sin(theta) -h/2*torch.cos(theta))
    edge3 = (x +w/2*torch.cos(theta) -h/2*torch.sin(theta), y +w/2*torch.sin(theta) +h/2*torch.cos(theta))
    edge4 = (x -w/2*torch.cos(theta) -h/2*torch.sin(theta), y -w/2*torch.sin(theta) +h/2*torch.cos(theta))
    return [edge1, edge2, edge3, edge4]