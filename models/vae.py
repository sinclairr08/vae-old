import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image

from utils import to_gpu, log_line

class VAE(nn.Module):
    def __init__(self, nlatent, ninput, nhidden, is_gpu):

        super(VAE, self).__init__()
        # self init area
        self.is_gpu = is_gpu
        self.nlatent = nlatent
        self.ninput = ninput
        self.nhidden = nhidden

        # mod : Modlarize within the class
        # self.enc = None
        # self.dec = None

        # Temp code
        self.enc1 = nn.Linear(self.ninput, self.nhidden)
        self.enc21 = nn.Linear(self.nhidden, self.nlatent)
        self.enc22 = nn.Linear(self.nhidden, self.nlatent)
        self.dec1 = nn.Linear(self.nlatent, self.nhidden)
        self.dec2 = nn.Linear(self.nhidden, self.ninput)

    # mod : Modularization
    # mod : layer of mu, logvar
    def encode(self, input):
        h1 = F.relu(self.enc1(input.view(-1, self.ninput)))
        return self.enc21(h1), self.enc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = to_gpu(torch.rand_like(std), self.is_gpu)  # epsilon for reparametrization
        return mu + eps * std

    # mod : Modularization
    def decode(self, latent):
        h2 = F.relu(self.dec1(latent))
        return torch.sigmoid(self.dec2(h2))

    # Returns the input to the output, mu, and log var
    def forward(self, input):
        mu, logvar = self.encode(input)
        latent = self.reparameterize(mu, logvar)
        output = self.decode(latent)

        return output, mu, logvar

    def loss(self, input, output, mu, logvar):
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        nll_loss = loss_fn(output, input.view(-1, self.ninput))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))

        return nll_loss + kl_loss

    def train_epoch(self, epoch, optimizer, train_loader, log_file):
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

        log_line("Epoch {} Train Loss : {:.4f}".format(
            epoch, total_loss / len(train_loader.dataset)), log_file, is_print=True, is_line=False)

    def test_epoch(self, epoch, test_loader, log_file):
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = to_gpu(data, self.is_gpu)
                output, mu, logvar = self.forward(data)
                loss = self.loss(data, output, mu, logvar)
                total_loss += loss.item()

        log_line("Test Loss : {:.4f}".format(
            total_loss / len(test_loader.dataset)), log_file, is_print=True)

    def sample(self, epoch, sample_num, save_path):
        with torch.no_grad():
            random_noise = to_gpu(torch.randn(sample_num, self.nlatent), self.is_gpu)
            sample = self.decode(random_noise).cpu()
            save_image(sample.view(sample_num, 1, 28, 28),
                       save_path + 'epoch_' + str(epoch) + '.png')