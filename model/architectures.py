import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_batch_by_center(x, shape):
    """
    Crop target area from x image tensor by new shape, shape[:-2] < x.shape[:-2]
    Args:
        x: input image 4-D tensor
        shape: result shape

    Returns:
        cropped image tensor
    """
    target_shape = shape[-2:]
    input_tensor_shape = x.shape[-2:]

    crop_by_y = (input_tensor_shape[0] - target_shape[0]) // 2
    crop_by_x = (input_tensor_shape[1] - target_shape[1]) // 2

    indexes_by_y = (
        crop_by_y, input_tensor_shape[0] - crop_by_y
    )

    indexes_by_x = (
        crop_by_x, input_tensor_shape[1] - crop_by_x
    )

    return x[
           :,
           :,
           indexes_by_y[0]:indexes_by_y[1],
           indexes_by_x[0]:indexes_by_x[1]
           ]


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
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
        x4 = self.fc2(x4)
        return F.softmax(x4, dim=1), x1, x3


class ConfidenceAE(nn.Module):
    def __init__(self, basic_net):
        super(ConfidenceAE, self).__init__()

        self.basic_net = basic_net
        self.basic_net.eval()

        self.conv1 = nn.Conv2d(20, 50, 4, 1)
        self.conv2 = nn.Conv2d(50, 50, 2, 1)
        self.conv3 = nn.Conv2d(100, 20, 1, 1)
        self.conv4 = nn.Conv2d(20, 10, 3, 1, padding=4)
        self.conv5 = nn.Conv2d(10, 1, 1, 1)

    def forward(self, x):
        _, x1, x2 = self.basic_net(x)
        x1 = x1.detach()
        x2 = x2.detach()

        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.max_pool2d(x1, 2, 2)
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = torch.sigmoid(self.conv5(x))
        return x
