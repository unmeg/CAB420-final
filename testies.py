import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torch.autograd import Variable

import time

class Testies(object):
    """
        Args:
            net (:class:`torch.nn.Module`) network to use
            dataset (:class: `np.array`)
            test_percent (float)
            starting_epoch (int)
            num_epochs (int)
    """
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

        self.net = net
        self.dataset = dataset
        self.test_percent = test_percent
        self.learning_rate = learning_rate

        self.starting_epoch = starting_epoch
        self.num_epochs = num_epochs

        self.generate_dataloaders()

        self.optimizer = optimizer or optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        self.loss_function = loss_function or nn.CrossEntropyLoss()

        self.dtype = torch.FloatTensor
        self.num_gpus = torch.cuda.device_count()

        # Check how many GPUs, do cuda/DataParallel accordingly
        if self.num_gpus > 0:
            self.dtype = torch.cuda.FloatTensor
            self.net.cuda()

        if self.num_gpus > 1:
            self.net = nn.DataParallel(self.net).cuda()


    def generate_dataloaders(self):

        rand_idxs = torch.randperm(len(self.dataset))
        test_size = int(np.floor(len(self.dataset) * self.test_percent))

        test_idxs = rand_idxs[:test_size]
        train_idxs = rand_idxs[test_size:]

        self.train_dl = data.DataLoader(
            self.dataset,
            sampler=data.sampler.SubsetRandomSampler(train_idxs),
            batch_size=1,
            num_workers=0
        )

        self.test_dl = data.DataLoader(
            self.dataset,
            sampler=data.sampler.SubsetRandomSampler(test_idxs),
            batch_size=1,
            num_workers=0
        )

    def train(self):
        # turn on training mode
        self.net.train()
        loss_log = []

        for i, (x, y) in enumerate(self.train_dl):
            if self.num_gpus > 0:
                x_var = x.cuda(non_blocking=True)
                y_var = y.cuda(non_blocking=True).type(torch.cuda.LongTensor)
            else:
                x_var = Variable(x.type(torch.FloatTensor))
                y_var = Variable(y.type(torch.LongTensor))

            # Forward pass
            out = self.net(x_var)
            # Compute loss
            # print(y_var.data)
            loss = self.loss_function(out, y_var)
            loss_log.append(loss.item())
            # Zero gradients before the backward pass
            self.optimizer.zero_grad()
            # Backprop
            loss.backward()
            # Update the params
            self.optimizer.step()

            # print(loss.item())
        return loss.item()

    def test(self):
        self.net.eval() # eval mode

        correct = 0
        total = 0

        for x, y in self.test_dl:
            if self.num_gpus > 0:
                x_var = Variable(x).cuda(non_blocking=True)
                y = y.type(torch.cuda.LongTensor)
            else:
                x_var = Variable(x.type(self.dtype))
                y = y.type(torch.LongTensor)

            outputs = self.net(x_var)
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted.cpu() == y).sum()

        accuracy = 100 * correct / total
        return accuracy

    def eval(self):
        best_epoch = 0
        best_accuracy = 0

        start_time = time.time()

        for epoch in range(self.starting_epoch, self.num_epochs):
            try:
                loss_log = self.train()
                print('Epoch {}/{} training loss: {}%'.format(epoch, self.num_epochs, loss_log))
                accuracy = self.test()
                print('Epoch {}/{} validation accuracy: {}%'.format(epoch, self.num_epochs, accuracy))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    torch.save(self.net.state_dict(), 'best_model.pkl')

            except KeyboardInterrupt: # Allow loop breakage
                print('\nBest accuracy of {}% at epoch {}\n'.format(best_accuracy, best_epoch))
                break
        time_taken = time.time() - start_time
        print('\nBest accuracy of {}% at epoch {}/{} in {} seconds'.format(best_accuracy, best_epoch, self.num_epochs, time_taken))
