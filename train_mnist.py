import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from dae.dae import DAE
from beta_vae.beta_vae import BetaVAE
from history import History
import utils

# hyperparameters
num_epochs = 10000
batch_size = 128
lr = 1e-4
beta = 4
save_iter = 200

shape = (28, 28)
n_obs = shape[0] * shape[1]

# create DAE and ß-VAE and their training history
dae = DAE(n_obs, num_epochs, batch_size, 1e-3, save_iter, shape)
beta_vae = BetaVAE(n_obs, num_epochs, batch_size, 1e-4, beta, save_iter, shape)
history = History()

# fill autoencoder training history with examples
print('Filling history...', end='', flush=True)

transformation = transforms.Compose([
    transforms.ColorJitter(),
    transforms.ToTensor()
])

dataset = MNIST('data', transform=transformation, download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for data in dataloader:
    img, _ = data
    img = img.view(img.size(0), -1).numpy().tolist()
    history.store(img)
print('DONE')

model_dir = "model_checkpoints"
utils.create_dir(model_dir)

# train DAE
dae.train(history, model_dir)

# train ß-VAE
beta_vae.train(history, dae, model_dir)