"""
In this file I check if neural network is able to learn a kinematic model.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import random


def normalize(x):
    assert len(x.shape) == 1, "x is not a vector"
    norm = np.sqrt(x.dot(x))
    return x / norm


WIDTH = 200
HEIGHT = 200
IMG_SIZE = (WIDTH, HEIGHT, 3)


class KinematicModel(object):
    """
    A model of ball bouncing around in a box.
    """
    def __init__(self):
        self.x   = np.zeros(2, dtype=np.float32)
        self.yaw = normalize(np.array([2.0, 1.0]))
        self.v   = 5.0

        self.left_bound = -100.0
        self.right_bound = 100.0

        self.lower_bound = -100.0
        self.upper_bound = 100.0

    def flip_around_upper_bound(self):
        self.x[1] = -self.x[1] + 2 * self.upper_bound
        self.yaw[1] = -self.yaw[1]

    def flip_around_lower_bound(self):
        self.x[1] = -self.x[1] + 2 * self.lower_bound
        self.yaw[1] = -self.yaw[1]

    def flip_around_left_bound(self):
        self.x[0] = -self.x[0] + 2 * self.left_bound
        self.yaw[0] = -self.yaw[0]

    def flip_around_right_bound(self):
        self.x[0] = -self.x[0] + 2 * self.right_bound
        self.yaw[0] = -self.yaw[0]

    def get_full_state(self):
        return np.hstack([
            self.x,
            self.v * self.yaw
        ])

    def move(self, dt):
        self.x += self.v * self.yaw * dt

        while True:
            if self.x[0] >= self.right_bound:
                self.flip_around_right_bound()
            elif self.x[0] <= self.left_bound:
                self.flip_around_left_bound()
            elif self.x[1] <= self.lower_bound:
                self.flip_around_lower_bound()
            elif self.x[1] >= self.upper_bound:
                self.flip_around_upper_bound()
            else:
                break

    R = 5

    def pic(self):
        img = np.full(IMG_SIZE, 255, dtype=np.uint8)

        xx, yy = np.meshgrid(range(HEIGHT), range(WIDTH))
        selector = (yy - self.x[1] + self.left_bound) ** 2 + (xx - self.x[0] + self.lower_bound) ** 2 < 5 ** 2
        img[selector] = (155, 7, 7)

        return img


def visualize_model():
    model = KinematicModel()
    img = model.pic()

    plt.ion()
    fig, ax = plt.subplots()

    im = ax.imshow(img)
    fig.show()


    while True:
        model.move(1.0)
        img = model.pic()

        im.set_data(img)
        fig.canvas.draw()
        plt.pause(0.0001)


N_DATA_POINTS = 100000
dt = 0.1
IN_SIZE = 4
OUT_SIZE = 4

def gather_data():
    # Initialize gathering
    model = KinematicModel()
    all_data = []

    for _ in range(N_DATA_POINTS):
        x = model.get_full_state()
        model.move(dt)
        y = model.get_full_state()
        all_data.append((x, y))

    return all_data

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(IN_SIZE, 10)
        self.fc2 = nn.Linear(10, 64)
        self.fc3 = nn.Linear(64, 20)
        self.fc4 = nn.Linear(20, OUT_SIZE)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

BATCH_SIZE = 128
NUM_EPOCHS = 1000
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    # visualize_model()
    data = gather_data()
    model = NeuralNet()
    if use_cuda:
        model = model.cuda()
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters())

    # do the training
    # for i in range(NUM_EPOCHS):
    i = 0
    while True:
        i += 1
        sample = random.sample(data, BATCH_SIZE)
        X, y = zip(*sample)
        X, y = map(FloatTensor, (X, y))
        X, y = map(Variable, (X, y))
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        print("Epoch:\t{}\tLoss:\t{}".format(i, loss.data.cpu()[0]))

        # print("#" * 30)
        # print("Epoch:\t{}\nLoss:\t{}".format(i, loss.data.cpu()[0]))
        # print("#" * 30)
        # print()
        opt.step()
