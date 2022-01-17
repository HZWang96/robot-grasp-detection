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
        #### TODO: Write the code for the model architecture below ####
        
        ################################################################
        device = torch.device("cuda:"+str(opt.which_gpu) if torch.cuda.is_available() else "cpu")
        print(device)

        net = nn.Sequential(

                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),

                nn.Linear(12544, 512), nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(512, 5))

        self.model = net.to(device)

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, input):
        opt = opts().init()
        device = torch.device("cuda:"+str(opt.which_gpu) if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        output = self.model(input)
        return output

    def compute_loss(self, xc, yc):
        loss_sum = 0.0
        loss_smallest_sum = 0.0
        X_loss_sum = 0.0
        Y_loss_sum = 0.0
        Theta_loss_sum = 0.0
        Length_loss_sum = 0.0
        Width_loss_sum = 0.0
        loss_smallest_avg = 0.0
        X_loss_avg = 0.0
        Y_loss_avg = 0.0
        Theta_loss_avg = 0.0
        Length_loss_avg = 0.0
        Width_loss_avg = 0.0

        opt = opts().init()

        output1 = self(xc) # output from ResNet50
        # print('output from ResNet50:', output1)

        for i in range(np.shape(output1)[0]):
            #Post_Processing the Output Value from the Model!!!!!
            x_pred, y_pred, theta_pred_1, length_pred, width_pred = output1[i]
            theta_pred = 0.5*torch.atan(2*theta_pred_1)

            Loss_Sum = []

            for z in range(np.shape(yc[i])[0]):
                x, y, theta, length, width = yc[i][z]      #第一个i是指第一个tensor, 第二个z是指tensor中的第z行

                # MSE Loss
                x_loss = F.mse_loss(x_pred, x)
                y_loss = F.mse_loss(y_pred, y)
                theta_loss = F.mse_loss(theta_pred, theta)
                length_loss = F.mse_loss(length_pred, length)
                width_loss = F.mse_loss(width_pred, width)

                # #L1 Loss
                # L1loss = nn.L1Loss()
                # x_loss = L1loss(x_pred, x)
                # y_loss = L1loss(y_pred, y)
                # theta_loss = L1loss(theta_pred, theta)
                # length_loss = L1loss(length_pred, length)
                # width_loss = L1loss(width_pred, width)

                # # Smooth L1 Loss
                # Smooth_L1loss = nn.SmoothL1Loss()
                # x_loss = Smooth_L1loss(x_pred, x)
                # y_loss = Smooth_L1loss(y_pred, y)
                # theta_loss = Smooth_L1loss(theta_pred, theta)
                # length_loss = Smooth_L1loss(length_pred, length)
                # width_loss = Smooth_L1loss(width_pred, width)

                gamma = torch.tensor(1.0)                   #对每个参数进行权重的调整
                alpha = torch.tensor(1.0)
                beta = torch.tensor(1.0)

                loss_sum = alpha*x_loss + alpha*y_loss + gamma*theta_loss + beta*length_loss + beta*width_loss
                Loss_Sum.append(loss_sum)

            loss_smallest = min(Loss_Sum)
            idx = Loss_Sum.index(min(Loss_Sum))
            x1, y1, theta1, length1, width1 = yc[i][idx]

            # MSE Loss
            X_loss = F.mse_loss(x_pred, x1)
            Y_loss = F.mse_loss(y_pred, y1)
            Theta_loss = F.mse_loss(theta_pred, theta1)
            Length_loss = F.mse_loss(length_pred, length1)
            Width_loss = F.mse_loss(width_pred, width1)
        
            # # L1 Loss
            # X_loss = L1loss(x_pred, x1)
            # Y_loss = L1loss(y_pred, y1)
            # Theta_loss = L1loss(theta_pred, theta1)
            # Length_loss = L1loss(length_pred, length1)
            # Width_loss = L1loss(width_pred, width1)

            # # SmoothL1Loss
            # X_loss = Smooth_L1loss(x_pred, x1)
            # Y_loss = Smooth_L1loss(y_pred, y1)
            # Theta_loss = Smooth_L1loss(theta_pred, theta1)
            # Length_loss = Smooth_L1loss(length_pred, length1)
            # Width_loss = Smooth_L1loss(width_pred, width1)

            loss_smallest_sum += loss_smallest
            X_loss_sum += X_loss
            Y_loss_sum += Y_loss
            Theta_loss_sum += Theta_loss
            Length_loss_sum += Length_loss
            Width_loss_sum += Width_loss

        loss_smallest_avg = loss_smallest_sum/opt.batch_size
        X_loss_avg = X_loss_sum/opt.batch_size
        Y_loss_avg = Y_loss_sum/opt.batch_size
        Theta_loss_avg = Theta_loss_sum/opt.batch_size
        Length_loss_avg = Length_loss_sum/opt.batch_size
        Width_loss_avg = Width_loss_sum/opt.batch_size

        return {
            'loss': loss_smallest_avg,                                                      #x_loss + y_loss + gamma*theta_loss + length_loss + width_loss,
            'losses': {
                'x_loss': X_loss_avg,
                'y_loss': Y_loss_avg,
                'theta_loss': Theta_loss_avg,
                'length_loss': Length_loss_avg,
                'width_loss': Width_loss_avg
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


def get_grasp_alexnet():
    
    alexnet = GraspNet()
    print("Initializing weights...")
    alexnet.apply(initNetParams)
    print("Weights are initialized! AlexNet model (Redmon version) loaded successfully!")

    return alexnet