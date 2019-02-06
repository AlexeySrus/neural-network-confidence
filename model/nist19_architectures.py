import torch
import torch.nn as nn
import torch.nn.functional as F


class NIST19Net(nn.Module):
    def __init__(self, classes, for_ae=False):
        super(NIST19Net, self).__init__()
        self.ae = for_ae

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(15*15*50, 500)
        self.fc2 = nn.Linear(500, classes)
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
        x3v = x3.view(-1, 15*15*50)
        x4 = F.relu(self.fc1(x3v))
        feat = x4
        x4 = self.fc2(x4)

        if self.ae:
            return F.softmax(x4, dim=1), feat
        return F.softmax(x4, dim=1)


class NIST19Net2(nn.Module):
    def __init__(self, classes, for_ae=False):
        super(NIST19Net2, self).__init__()
        self.ae = for_ae

        self.conv1 = nn.Conv2d(1, 40, 5, 1)
        self.conv2 = nn.Conv2d(40, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 20, 5, 1)
        self.fc1 = nn.Linear(20*14*14, 1500)
        self.fc2 = nn.Linear(1500, 500)
        self.fc3 = nn.Linear(500, classes)
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(40)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        if self.training:
            self.dropout(x)

        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        if self.training:
            self.dropout(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.bn3(x)
        x = F.relu(self.conv3(x))
        if self.training:
            self.dropout(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 20*14*14)
        x = F.relu(self.fc1(x))
        feat = x
        if self.training:
            self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.ae:
            return F.softmax(x, dim=1), feat
        return F.softmax(x, dim=1)


class ConfidenceAE(nn.Module):
    def __init__(self, basic_net):
        super(ConfidenceAE, self).__init__()

        self.basic_net = basic_net
        self.basic_net.eval()

        for p in self.basic_net.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(500, 2500)
        self.fc2 = nn.Linear(2500, 72*72)

    def forward(self, x):
        _, x1 = self.basic_net(x)
        x = F.relu(self.fc1(x1))
        x = torch.sigmoid(self.fc2(x))
        x = x.view(-1, 1, 72, 72)
        return x
