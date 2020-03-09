import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from dae.model import Model
from dae.visualize import *

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import utils

class DAE():
    def __init__(self, n_obs, num_epochs, batch_size, lr, save_iter, shape):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_iter = save_iter
        self.save_model_iter = 5 * self.save_iter
        self.shape = shape

        self.dae = Model(n_obs)

    def encode(self, x):
        return self.dae.encode(x)

    def decode(self, z):
        return self.dae.decode(z)

    def train(self, history, model_dir):
        print('Training DAE...', end='', flush=True)

        optimizer = optim.Adam(self.dae.parameters(), lr=self.lr)

        dae_model_dir = os.path.join(model_dir, "dae")
        utils.create_dir(dae_model_dir)

        for epoch in range(self.num_epochs):

            minibatches = history.get_minibatches(self.batch_size)
            for data in minibatches:

                out = self.dae(data)

                # calculate loss and update network
                loss = torch.pow(data - out, 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch == 0 or epoch % self.save_iter == self.save_iter - 1:
                pic = out.data.view(out.size(0), 1, self.shape[0], self.shape[1])
                save_image(pic, 'img/dae_' + str(epoch+1) + '_epochs.png')
            
            if epoch % self.save_model_iter == 0:
                checkpoint_path = "dae_model_{}_epochs".format(epoch)
                checkpoint_path = os.path.join(dae_model_dir, checkpoint_path)
                utils.save_checkpoint(Model, self.dae, checkpoint_path)

            # plot loss
            update_viz(epoch, loss.item())

        print('DONE')
