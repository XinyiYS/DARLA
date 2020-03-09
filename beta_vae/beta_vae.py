import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from beta_vae.model import Model
from beta_vae.visualize import *

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import utils

class BetaVAE():
    def __init__(self, n_obs, num_epochs, batch_size, lr, beta, save_iter, shape):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.save_iter = save_iter
        self.save_model_iter = 5 * self.save_iter
        self.shape = shape

        self.n_obs = n_obs

        self.vae = Model(n_obs)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def train(self, history, dae, model_dir):
        print('Training ß-VAE...', end='', flush=True)

        def KL(mu, log_var):
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl /= mu.size(0) * self.n_obs
            return kl

        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)

        betaVae_model_dir = os.path.join(model_dir, "betaVae")
        utils.create_dir(betaVae_model_dir)

        for epoch in range(self.num_epochs):

            minibatches = history.get_minibatches(self.batch_size)
            for data in minibatches:

                out, mu, log_var = self.vae(data)

                # calculate loss and update network
                loss = torch.pow(dae.encode(data) - dae.encode(out), 2).mean() + (self.beta * KL(mu, log_var))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch == 0 or epoch % self.save_iter == self.save_iter - 1:
                pic = out.data.view(out.size(0), 1, self.shape[0], self.shape[1])
                save_image(pic, 'img/betaVae_' + str(epoch+1) + '_epochs.png')

            if epoch % self.save_model_iter == 0:
                checkpoint_path = "betaVae_model_{}_epochs".format(epoch)
                checkpoint_path = os.path.join(betaVae_model_dir, checkpoint_path)
                utils.save_checkpoint(Model, self.vae, checkpoint_path)
            
            # plot loss
            update_viz(epoch, loss.item())

        print('DONE')
