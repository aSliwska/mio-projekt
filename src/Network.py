import torch.nn as nn
import torch

class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        self.inp = nn.Linear(6, 16)
        self.conv1 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(576, 288)
        self.fc2 = nn.Linear(288, 72)
        self.fc3 = nn.Linear(72, 10)
        self.outp = nn.Linear(32, 4)  # 4 output classes

    def forward(self, x):
        x = F.relu(self.inp(x))
        x = F.relu(self.pool(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.outp(x)
        return x