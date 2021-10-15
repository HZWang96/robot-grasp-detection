import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torchsummary import summary
# from d2l import torch as d2l
from torchvision import datasets, models, transforms
import torch.nn as nn
from opts import opts
import numpy as np

class GraspNet(nn.Module):
    def __init__(self):
        opt = opts().init()
        super().__init__()
        
        device = torch.device("cuda:"+str(opt.which_gpu) if torch.cuda.is_available() else "cpu")
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
        opt = opts().init()
        device = torch.device("cuda:"+str(opt.which_gpu) if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        output = self.model(input)
        return output[0][0], output[0][1], output[0][2], output[0][3], output[0][4]

    def compute_loss(self, xc, yc):
        loss_sum = 0.0
        pred = []
        Loss_Sum = []
        opt = opts().init()
        # print(yc)
        # print(np.shape(yc))
        # print(np.shape(yc[0]))
        # print(np.shape(yc[0])[0])
        for i in range(opt.batch_size)
            for z in range(np.shape(yc[i])[0]):
                x, y, theta, length, width = yc[z]       # z是指tensor中的第z行
                x_pred, y_pred, theta_pred, length_pred, width_pred = self(xc)
                pred.append((x_pred, y_pred, theta_pred, length_pred, width_pred))

                x_loss = F.mse_loss(x_pred, x)
                y_loss = F.mse_loss(y_pred, y)
                theta_loss = F.mse_loss(theta_pred, theta)
                length_loss = F.mse_loss(length_pred, length)
                width_loss = F.mse_loss(width_pred, width)
                gamma = torch.tensor(10.)

                loss_sum = x_loss + y_loss + gamma*theta_loss + length_loss + width_loss
                Loss_Sum.append(loss_sum)

            # Loss_Sum = Loss_Sum.tolist()
            loss_smallest = min(Loss_Sum)
            idx = Loss_Sum.index(min(Loss_Sum))
            x1, y1, theta1, length1, width1 = yc[idx]
            X_pred = pred[idx][0]                           #Careful Dropout!!!
            Y_pred = pred[idx][1]
            Theta_pred = pred[idx][2]
            Length_pred = pred[idx][3]
            Width_pred = pred[idx][4]

            X_loss = F.mse_loss(X_pred, x1)
            Y_loss = F.mse_loss(Y_pred, y1)
            Theta_loss = F.mse_loss(Theta_pred, theta1)
            Length_loss = F.mse_loss(Length_pred, length1)
            Width_loss = F.mse_loss(Width_pred, width1)

            # if z=0:
            #     loss_sum2 = loss_sum1
            # else:
            #     if loss_sum1<loss_sum2:
            #             loss_sum2 = loss_sum1
            #             idx += 1
            #     else:

        # x, y, theta, length, width = yc[0][0]       #第一个0是指第一个tensor, 第二个0是指tensor中的第一行
        # x_pred, y_pred, theta_pred, length_pred, width_pred = self(xc)

        # x_loss = F.mse_loss(x_pred, x)
        # y_loss = F.mse_loss(y_pred, y)
        # theta_loss = F.mse_loss(theta_pred, theta)
        # length_loss = F.mse_loss(length_pred, length)
        # width_loss = F.mse_loss(width_pred, width)
        # gamma = torch.tensor(10.)
        # print(theta_loss.type())
        # print("x_pred:",x_pred, "y_pred:",y_pred, "theta_pred:",theta_pred, "length_pred:",length_pred, "width_pred:",width_pred)
        # print("x:",x, "y:",y, "theta:",theta, "length:",length, "width:",width)

        return {
            'loss': loss_smallest,                                                      #x_loss + y_loss + gamma*theta_loss + length_loss + width_loss,
            'losses': {
                'x_loss': X_loss,
                'y_loss': Y_loss,
                'theta_loss': Theta_loss,
                'length_loss': Length_loss,
                'width_loss': Width_loss
            },
            'pred': {
                'x': X_pred,
                'y': Y_pred,
                'theta': Theta_pred,
                'length': Length_pred,
                'width': Width_pred
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