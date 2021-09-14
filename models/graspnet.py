import torch
from torch import nn
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


def get_graspnet():
    
    model = GraspNet()

    return model