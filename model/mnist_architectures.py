import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self, for_ae=False, n_classes=10):
        super(MNISTNet, self).__init__()
        self.ae = for_ae

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, n_classes)
        self.dropoutcv1 = nn.Dropout2d(0.1)
        self.dropoutcv2 = nn.Dropout2d(0.1)
        self.dropoutln = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropoutcv1(x)
        x1 = F.max_pool2d(x, 2, 2)
        x2 = F.relu(self.conv2(x1))
        x2 = self.dropoutcv2(x2)
        x3 = F.max_pool2d(x2, 2, 2)
        x3v = x3.view(-1, 4*4*50)
        x4 = F.relu(self.fc1(x3v))
        feat = x3
        x4 = self.dropoutln(x4)
        x4 = self.fc2(x4)

        if self.ae:
            return F.softmax(x4, dim=1), feat
        return F.softmax(x4, dim=1)

    def inference(self, x):
        x = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x, 2, 2)
        x2 = F.relu(self.conv2(x1))
        x3 = F.max_pool2d(x2, 2, 2)
        x3v = x3.view(-1, 4 * 4 * 50)
        x4 = F.relu(self.fc1(x3v))
        feat = x3
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

        self.deconv1 = nn.ConvTranspose2d(50, 30, 6)
        self.deconv2 = nn.ConvTranspose2d(30, 15, 10)
        self.deconv3 = nn.ConvTranspose2d(15, 5, 10)
        self.deconv4 = nn.ConvTranspose2d(5, 1, 4)
        self.conv1 = nn.Conv2d(1, 1, 3)


    def forward(self, x):
        _, x1 = self.basic_net.inference(x)

        x = self.deconv1(x1)
        x = nn.functional.relu(x)
        x = self.deconv2(x)
        x = nn.functional.relu(x)
        x = self.deconv3(x)
        x = nn.functional.relu(x)
        x = self.deconv4(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x

    def inference(self, x):
        _, x1 = self.basic_net.inference(x)

        x = self.deconv1(x1)
        x = nn.functional.relu(x)
        x = self.deconv2(x)
        x = nn.functional.relu(x)
        x = self.deconv3(x)
        x = nn.functional.relu(x)
        x = self.deconv4(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x
