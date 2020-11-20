import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionNet(nn.Module):
    def __init__(self):
        super(PerceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 36, 5, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(36)

        self.conv3 = nn.Conv2d(36, 48, 5, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48, 64, 3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 21 * 21, 100, bias=False)
        self.fc2 = nn.Linear(100, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.output = nn.Linear(10, 1, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.output(x))


if __name__ == '__main__':
    net = PerceptionNet()
    input = torch.randn((32, 3, 300, 400))
    output = net(input)
    print(output.shape)
