import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionNet(nn.Module):
    def __init__(self):
        super(PerceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 100, bias=False)
        self.fc2 = nn.Linear(100, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.output = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(self.pool(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.output(x))


if __name__ == '__main__':
    net = PerceptionNet()
    input = torch.randn((32, 3, 300, 400))
    output = net(input)
    print(output.shape)
