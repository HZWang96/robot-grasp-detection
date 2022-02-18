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

        self.model = models.resnet18(pretrained=False).to(device)

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 5), 
        nn.ReLU(inplace=True))

        # nn.Linear(512, 512), 
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.2),
        # nn.Linear(512, 5))

    def forward(self, input):
        opt = opts().init()
        device = torch.device("cuda:"+str(opt.which_gpu) if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        output = self.model(input)
        # print("Printing output from ResNet-50:")
        # print(output)
        # return output[0][0], output[0][1], output[0][2], output[0][3], output[0][4]
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

        # pred = []
        opt = opts().init()
        # print(yc)
        # print(np.shape(yc))
        # print(np.shape(yc[0]))
        # print(np.shape(yc[0])[0])
        # print(xc.shape)
        # print(self(xc[1].unsqueeze(0)))
        # print("Printing output from ResNet50:")
        # print(self(xc))
        output1 = self(xc) # output from ResNet50
        # print('output from ResNet50:', output1)

        for i in range(np.shape(output1)[0]):
            # x_pred, y_pred, theta_pred, length_pred, width_pred = self(xc[i].unsqueeze(0))   #不能放在第二个for循环里面
            # x_pred, y_pred, theta_pred, length_pred, width_pred = output1[i]

            #Post_Processing the Output Value from the Model!!!!!
            # x_pred_1, y_pred_1, sintheta_pred, costheta_pred, length_pred_1, width_pred_1 = output1[i]
            # x_pred, y_pred, sintheta_pred, costheta_pred, length_pred, width_pred = output1[i]
            # x_pred_1, y_pred_1, theta_pred_1, length_pred_1, width_pred_1 = output1[i]
            x_pred, y_pred, theta_pred_1, length_pred, width_pred = output1[i]
            # print('Theta from model is:', theta_pred_1)

            # x_pred = x_pred_1  / 224
            # y_pred = y_pred_1  / 224
            # length_pred = length_pred_1  / 100
            # width_pred = width_pred_1  / 80

            # sin2theta_pred = torch.sin(2*theta_pred_1)
            # cos2theta_pred = torch.cos(2*theta_pred_1)

            # tan2theta_pred = torch.div(sin2theta_pred, cos2theta_pred)
            # theta_pred = 0.5*torch.atan(tan2theta_pred)
            # theta_pred = 0.5*torch.atan2(sin2theta_pred, cos2theta_pred) #careful torch.atan and torch.atan2!!!
            theta_pred = 0.5*torch.atan(2*theta_pred_1)
            # theta_pred = 0.5*torch.atan(theta_pred_1)
            # print('theta_pred is:', theta_pred)

            Loss_Sum = []
            # pred.append((x_pred, y_pred, theta_pred, length_pred, width_pred))

            for z in range(np.shape(yc[i])[0]):
                x, y, theta, length, width = yc[i][z]      #第一个i是指第一个tensor, 第二个z是指tensor中的第z行
                # print('Theta from GT is:', theta)

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

            # Loss_Sum = Loss_Sum.tolist()
            loss_smallest = min(Loss_Sum)
            idx = Loss_Sum.index(min(Loss_Sum))
            x1, y1, theta1, length1, width1 = yc[i][idx]
            # X_pred = pred[idx][0]                           #Careful Dropout!!!
            # Y_pred = pred[idx][1]
            # Theta_pred = pred[idx][2]
            # Length_pred = pred[idx][3]
            # Width_pred = pred[idx][4]

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


def get_grasp_resnet():
    
    resnet_50 = GraspNet()
    print("Initializing weights...")
    resnet_50.apply(initNetParams)
    print("Weights are initialized! ResNet model loaded successfully!")

    return resnet_50