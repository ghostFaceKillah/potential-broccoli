"""
In this file I check if neural network is able to learn a kinematic model.
"""

# import matplotlib.pyplot as plt

import ipdb
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from torch.autograd import Variable


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
    def __init__(self, x, yaw):
        self.x = x
        self.yaw = normalize(yaw)
        self.v = 5.0

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

    def gather_run(self, how_much=256):
        xs = [self.x.copy()]

        for i in xrange(how_much + 1):
            self.move(0.1)
            xs.append(self.x.copy())

        return zip(xs[:-1], xs[1:])

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


HIDDEN_SIZE = 10

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2 + HIDDEN_SIZE, 10)
        self.fc2 = nn.Linear(10, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc_to_out = nn.Linear(64, 2)
        self.fc_to_hidden = nn.Linear(64, 10)

    def forward(self, x, hidden):
        x = torch.cat((x, hidden))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        out = self.fc_to_out(x)
        new_hidden = self.fc_to_hidden(x)
        return out, new_hidden

    def init_hidden(self):
        return Variable(torch.zeros(HIDDEN_SIZE))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    model = NeuralNet()
    if use_cuda:
        model = model.cuda()

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters())
    # opt = optim.RMSprop(model.parameters())
    # opt = optim.SGD(model.parameters(), lr=0.001)

    # do the training
    # for i in range(NUM_EPOCHS):
    i = 0
    while True:
        i += 1

        opt.zero_grad()
        # model.zero_grad()

        sp = np.array([
            np.random.rand() * WIDTH - 100,
            np.random.rand() * HEIGHT - 100
        ])
        v = np.random.rand(2)

        kin_model = KinematicModel(sp, v)
        trajectory = kin_model.gather_run()
        hidden = model.init_hidden()

        loss = 0
        for x, y in trajectory:
            x, y = Variable(FloatTensor(x)), Variable(FloatTensor(y))
            y_hat, hidden = model(x, hidden)
            loss += criterion(y_hat, y)

        loss.backward()
        opt.step()

        print("Epoch:\t{}\tLoss:\t{}".format(i, loss.data.cpu()[0]))
