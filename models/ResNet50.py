import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
# from d2l import torch as d2l
from torchvision import datasets, models, transforms
import torch.nn as nn

class GraspNet(nn.Module):
    def __init__(self):
        super().__init__()
        #### TODO: Write the code for the model architecture below ####
        
        ################################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        model = models.resnet50(pretrained=True).to(device)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 1024).to(device), nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 5))

    # def forward(self, x):
        #### TODO: Write the code for the model architecture below ####
        
        ################################################################
        # return x

# Model initialization
def initNetParams(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight, std=0.001)
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # init.normal(m.weight, std=0.001)
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias:
                init.constant(m.bias, 0)

# GraspNet.apply(initNetParams) # 加在init函数里面
# print("Weights are initialized!")


def get_graspnet():
    
    resnet_50 = GraspNet()
    resnet_50.apply(initNetParams) # 加在init函数里面
    print("Weights are initialized!")


    return resnet_50