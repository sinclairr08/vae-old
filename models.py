import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.utils import save_image
from utils import to_gpu

# Reference : https://github.com/pytorch/examples/tree/master/vae
# Have to add at the readme.md

class VAE(nn.Module):
    def __init__(self, nlatent, is_gpu):

        super(VAE, self).__init__()
        # self init area
        self.is_gpu = is_gpu
        self.nlatent = nlatent

        # NEED TO MODIFY THIS!! MODULARIZATION
        #self.enc = None
        #self.enc = None
        #self.dec = None

        # Temp code
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.nlatent)
        self.fc22 = nn.Linear(400, self.nlatent)
        self.fc3 = nn.Linear(self.nlatent, 400)
        self.fc4 = nn.Linear(400, 784)

    # MODIFICATION REQUIRED
    def encode(self, input):
        """
        if self.enc is None:
            raise NotImplementedError
        return self.enc(input)
        """

        h1 = F.relu(self.fc1(input))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std) # epsilon for reparametrization
        return mu + eps*std

    # MODIFICATION REQUIRED
    def decode(self, latent):
        """
        if self.dec is None:
            raise NotImplementedError
        return self.dec(latent)
        """
        h3 = F.relu(self.fc3(latent))
        return torch.sigmoid(self.fc4(h3))

    # Returns the input to the output, mu, and log var
    def forward(self, input):
        mu, logvar = self.encode(input.view(-1, 784))
        latent = self.reparameterize(mu, logvar)
        output = self.decode(latent)

        return output, mu, logvar

    def loss(self, input, output, mu, logvar):
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        recon_loss = loss_fn(output, input.view(-1, 784)) ## Need dimension modification > have to do something more ,,, Is it really sum??
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))

        return recon_loss + kl_loss

    def train_epoch(self, epoch, optimizer, train_loader):
        self.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = to_gpu(data, self.is_gpu)
            output, mu, logvar = self.forward(data)
            optimizer.zero_grad()

            loss = self.loss(data, output, mu, logvar)
            loss.backward()
            total_loss += loss.item()

            optimizer.step()

        print("Epoch {} Train Loss : {:.4f}".format(
            epoch, total_loss / len(train_loader.dataset)
        ), end = ' ')

    def test_epoch(self, epoch, test_loader):
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = to_gpu(data, self.is_gpu)
                output, mu, logvar = self.forward(data)
                loss = self.loss(data, output, mu, logvar)
                total_loss += loss.item()

        print("Test Loss : {:.4f}".format(
            total_loss / len(test_loader.dataset)
        ))

    def sample(self, epoch, sample_num, save_path): #### Latent information should be given
        with torch.no_grad():
            latent_sample = to_gpu(torch.randn(sample_num, self.nlatent), self.is_gpu)
            sample = self.decode(latent_sample).cpu()
            save_image(sample.view(sample_num, 1, 28, 28),               # Modify
                       save_path + 'epoch ' + str(epoch) + '.png')





















