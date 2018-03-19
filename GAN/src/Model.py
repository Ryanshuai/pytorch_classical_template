import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.defc = nn.Sequential(
            nn.Linear(128, 4*4*1024),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
            nn.ReLU()
        )

    def forward(self, input):
        x = self.defc(input)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(560, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 10),
            nn.BatchNorm1d(10),
            nn.LogSoftmax()
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = x.view(-1,560)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

