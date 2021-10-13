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
        return output[0][0], output[0][1], output[0][2], output[0][3], output[0][4]

    def compute_loss(self, xc, yc):
        # print(yc)
        x, y, theta, length, width = yc[0][0]       #第一个0是指第一个tensor, 第二个0是指tensor中的第一行
        x_pred, y_pred, theta_pred, length_pred, width_pred = self(xc)

        x_loss = F.mse_loss(x_pred, x)
        y_loss = F.mse_loss(y_pred, y)
        theta_loss = F.mse_loss(theta_pred, theta)
        length_loss = F.mse_loss(length_pred, length)
        width_loss = F.mse_loss(width_pred, width)
        gamma = torch.tensor(10.)
        # print(theta_loss.type())

        return {
            'loss': x_loss + y_loss + gamma*theta_loss + length_loss + width_loss,
            'losses': {
                'x_loss': x_loss,
                'y_loss': y_loss,
                'theta_loss': theta_loss,
                'length_loss': length_loss,
                'width_loss': width_loss
            },
            'pred': {
                'x': x_pred,
                'y': y_pred,
                'theta': theta_pred,
                'length': length_pred,
                'width': width_pred
            }
        }


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