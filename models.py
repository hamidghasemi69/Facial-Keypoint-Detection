## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.2)
        
        
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.3)
        
        
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.3)
        # output = (256, 10, 10)
        
        
        self.fc5 = nn.Linear(256*10*10, 5120)
        # dropout with p=0.4
        self.drop5 = nn.Dropout(p=0.4)
        
        self.fc6 = nn.Linear(5120, 1024)
        # dropout with p=0.3
        self.drop6 = nn.Dropout(p=0.3)
        
        
        # finally, create 10 output channels (for the 10 classes)
        self.fc7 = nn.Linear(1024, 136)


        
    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc5(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        
        x = self.fc7(x)
        
        
        return x
