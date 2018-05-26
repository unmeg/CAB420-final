import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torch.autograd import Variable


class Testies(object):
    def __init__(self,
        net,
        dataset,
        test_percent=0.3,
        learning_rate=1e-4,
        starting_epoch=0,
        num_epochs=50,
        optimizer=None,
        loss_function=None
    ):
        """
            Args:
              net (:class:`torch.nn.Module`) network to use
              dataset (:class: `np.array`)
              test_percent (float)
              starting_epoch (int)
              num_epochs (int)
        """
        self.net = net
        self.dataset = dataset
        self.test_percent = test_percent
        self.learning_rate = learning_rate

        self.starting_epoch = starting_epoch
        self.num_epochs = num_epochs

        self.generate_dataloaders()

        self.optimizer = optimizer or optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        self.loss_function = loss_function or nn.CrossEntropyLoss()

    def generate_dataloaders(self):
        rand_idxs = torch.randperm(len(self.dataset))
        test_size = int(np.floor(len(self.dataset) * self.test_percent))

        test_idxs = rand_idxs[:test_size]
        train_idxs = rand_idxs[test_size:]

        self.train_dl = data.DataLoader(
            self.dataset,
            sampler=data.sampler.SubsetRandomSampler(train_idxs),
            num_workers=1
        )
        print([x for x in self.train_dl])

        self.test_dl = data.DataLoader(
            self.dataset,
            sampler=data.sampler.SubsetRandomSampler(test_idxs),
            num_workers=1
        )

    def train(self):
        # turn on training mode
        self.net.train()
        loss_log = []

        for epoch in range(self.starting_epoch, self.num_epochs):

            for i, (x, y) in enumerate(self.train_dl):

                x_var = Variable(x.type(dtype))
                y_var = Variable(y.type(dtype))

                # Forward pass
                out = self.net(x_var)
                # Compute loss
                loss = self.loss_function(out, y_var)
                loss_log.append(loss.item())
                # Zero gradients before the backward pass
                self.optimizer.zero_grad()
                # Backprop
                loss.backward()
                # Update the params
                self.optimizer.step()
