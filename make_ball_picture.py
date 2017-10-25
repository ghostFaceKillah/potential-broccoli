import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 320
HEIGHT = 240

IMG_SIZE = (WIDTH, HEIGHT, 3)

COLOR = (7, 7, 155)


def make_img(center_x, center_y, r, color=COLOR):
    img = np.full(IMG_SIZE, 255)

    xx, yy = np.meshgrid(range(HEIGHT), range(WIDTH))
    selector = (xx - center_x) ** 2 + (yy - center_y) ** 2 < r ** 2
    img[selector] = color

    return img


class BallLocator(nn.Module):
    """
    Neural network that finds the ball in the picture
    """
    def __init__(self):
        super(BallLocator, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view())



center_x = 100
center_y = 150
r = 21




# cv2.imwrite('img.png', img)
