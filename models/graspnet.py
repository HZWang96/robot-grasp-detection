import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from d2l import torch as d2l

class GraspNet(nn.Module):
    def __init__(self):
        super().__init__()
        #### TODO: Write the code for the model architecture below ####
        
        ################################################################

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
            init.normal(m.weight, std=0.001)
            if m.bias:
                init.constant(m.bias, 0)

GraspNet.apply(initNetParams) # 加在init函数里面
print("Weights are initialized!")


def get_graspnet():
    
    model = GraspNet()

    return model