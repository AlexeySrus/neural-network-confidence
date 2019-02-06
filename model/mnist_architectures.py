import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self, for_ae=False):
        super(MNISTNet, self).__init__()
        self.ae = for_ae

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.training:
            self.dropout(x)
        x1 = F.max_pool2d(x, 2, 2)
        x2 = F.relu(self.conv2(x1))
        if self.training:
            self.dropout(x2)
        x3 = F.max_pool2d(x2, 2, 2)
        x3v = x3.view(-1, 4*4*50)
        x4 = F.relu(self.fc1(x3v))
        feat = x4
        x4 = self.fc2(x4)

        if self.ae:
            return F.softmax(x4, dim=1), feat
        return F.softmax(x4, dim=1)


class ConfidenceAE(nn.Module):
    def __init__(self, basic_net):
        super(ConfidenceAE, self).__init__()

        self.basic_net = basic_net
        self.basic_net.eval()

        for p in self.basic_net.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(500, 700)
        self.fc2 = nn.Linear(700, 28*28)

    def forward(self, x):
        _, x1 = self.basic_net(x)
        x = F.relu(self.fc1(x1))
        x = torch.sigmoid(self.fc2(x))
        x = x.view(-1, 1, 28, 28)
        return x
