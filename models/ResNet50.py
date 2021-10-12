import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torchsummary import summary
# from d2l import torch as d2l
from torchvision import datasets, models, transforms
import torch.nn as nn

class GraspNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        self.model = models.resnet50(pretrained=True).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 1024).to(device), nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 5))

    def forward(self, input):
        output = self.model(input)
        return output

# Model initialization
def initNetParams(layers):
    for m in layers.modules():
        # print(m)
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
            if m.bias is not None:
                init.constant(m.bias, 0)


def get_graspnet():
    
    resnet_50 = GraspNet()
    print("Initializing weights...")
    resnet_50.apply(initNetParams)
    print("Weights are initialized!")

    return resnet_50