import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

WIDTH = 100
HEIGHT = 50

IMG_SIZE = (WIDTH, HEIGHT, 3)

COLOR = (7, 7, 155)
R = 5
BATCH_SIZE = 64

use_cuda = torch.cuda.is_available()
if use_cuda:
    print "Cuda is available"
else:
    print "Cuda is not available"
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def make_img(center_y, center_x, r, color=COLOR):
    img = np.full(IMG_SIZE, 255)

    yy, xx = np.meshgrid(range(HEIGHT), range(WIDTH))
    selector = (yy - center_y) ** 2 + (xx - center_x) ** 2 < r ** 2
    img[selector] = color

    return img


def generate_sample():
    center_y = np.random.randint(HEIGHT)
    center_x = np.random.randint(WIDTH)
    img = make_img(center_y, center_x, R)

    img = torch.from_numpy(img)
    img = img.permute(2, 1, 0).unsqueeze(0)
    x = Variable(img).type(Tensor)

    y = Variable(Tensor([[center_x, center_y]]))

    return x, y




def generate_batch():
    xs, ys = [], []

    for _ in range(BATCH_SIZE):
        x, y = generate_sample()
        xs.append(x)
        ys.append(y)

    x_batch = torch.cat(xs).cuda()
    y_batch = torch.cat(ys).cuda()

    return x_batch, y_batch


class NeuralNet(nn.Module):
    """
    Ball locator
    Neural network that finds the ball in the picture
    """
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        

        self.fc1 = nn.Linear(132352, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 132352)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = NeuralNet()
if use_cuda:
    net.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

print "Initialized model, starting training"

epoch = 0

while True:
    epoch += 1

    xs, ys = generate_batch()
    optimizer.zero_grad()

    y_hat = net(xs)
    loss = criterion(y_hat, ys)
    loss.backward()
    optimizer.step()

    print (
            "Epoch {} ".format(epoch) +
            "loss = {} ".format(loss.cpu().data.numpy()) + 
            "y_hat = {} actual y = {}".format(y_hat[0].cpu().data.numpy(), ys[0].cpu().data.numpy())
    )

