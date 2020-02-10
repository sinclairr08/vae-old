import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

from utils import to_gpu, log_line

class AAE(nn.Module):
    def __init__(self, nlatent, ninput, nhidden, is_gpu):

        super(AAE, self).__init__()
        # self init area
        self.is_gpu = is_gpu
        self.nlatent = nlatent
        self.ninput = ninput
        self.nhidden = nhidden

        # mod : Modlarize within the class
        # self.enc = None
        # self.dec = None

        # Encoder

        self.encoder = nn.Sequential(
            nn.Linear(self.ninput, self.nhidden),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(self.nhidden, self.nhidden),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(self.nhidden, self.nlatent),
        )

        # Decoder

        self.decoder = nn.Sequential(
            nn.Linear(self.nlatent, self.nhidden),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(self.nhidden, self.nhidden),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(self.nhidden, self.ninput),
            nn.Sigmoid()
        )

        # Discriminator
        self.disc = nn.Sequential(
            nn.Linear(self.nlatent, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
            torch.nn.Sigmoid()
        )

        # mod : Total Optim requires somewhat change
        self.optim_enc_nll = torch.optim.Adam(self.encoder.parameters(), lr=1e-04)
        self.optim_enc_adv = torch.optim.Adam(self.encoder.parameters(), lr=5e-05)
        self.optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=1e-04)
        self.optim_disc = torch.optim.Adam(self.disc.parameters(), lr=5e-05)

        # Epsilon to prevent 0
        self.eps = 1e-15

    # mod : Check that is it really needs
    def encode(self, input):
        return self.encoder(input.view(-1, self.ninput))

    # mod : Check that is it really needs
    def decode(self, latent):
        return self.decoder(latent)

    # Returns the latent an doutput
    def forward(self, input):
        latent = self.encode(input)
        output = self.decode(latent)

        return output, latent

    def nll_loss(self, input, output):
        # loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        loss = F.binary_cross_entropy(output, input.view(-1, self.ninput))

        return loss

    def disc_loss(self, latent_real, latent_fake):
        assert latent_real.shape == latent_fake.shape

        disc_real = self.disc(latent_real)
        disc_fake = self.disc(latent_fake)

        loss = -torch.mean(torch.log(disc_real + self.eps) + torch.log(1 - disc_fake + self.eps))

        return loss

    def adv_loss(self, input):
        latent_fake = self.encode(input)
        disc_fake = self.disc(latent_fake)

        loss = -torch.mean(torch.log(disc_fake + self.eps))
        return loss

    def train_epoch(self, epoch, train_loader, log_file):
        self.train()

        total_nll_loss = 0
        total_adv_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = to_gpu(data, self.is_gpu)

            self.optim_enc_nll.zero_grad()
            self.optim_enc_adv.zero_grad()
            self.optim_dec.zero_grad()
            self.optim_disc.zero_grad()

            # Phase 1 : Train Autoencoder
            output, _ = self.forward(data)

            nll_loss = self.nll_loss(data, output)  # Need to check the names
            nll_loss.backward()
            total_nll_loss += nll_loss.item()
            self.optim_enc_nll.step()
            self.optim_dec.step()

            # Phase 2 : Train Discriminator
            latent_fake = self.encode(data)
            latent_real = to_gpu(Variable(torch.randn_like(latent_fake)), self.is_gpu)

            disc_loss = self.disc_loss(latent_real, latent_fake)
            disc_loss.backward()
            self.optim_disc.step()

            # Phase 3 : Train encoder
            adv_loss = self.adv_loss(data)
            adv_loss.backward()
            total_adv_loss += adv_loss.item()
            self.optim_enc_adv.step()

            # print("NLL Loss : {:.4f} ADV Loss : {:.4f} DISC Loss : {:.4f}".format(
            # nll_loss, adv_loss, disc_loss))

        total_loss = total_nll_loss + total_adv_loss
        len_data = len(train_loader.dataset)
        log_line("Epoch {} Train Loss : {:.4f} NLL Loss : {:.4f} Adv Loss : {:.4f}".format(
            epoch, total_loss / len_data, total_nll_loss / len_data,
                   total_adv_loss / len_data), log_file, is_print=True)

    def test_epoch(self, epoch, test_loader, log_file):
        self.eval()
        total_nll_loss = 0
        total_adv_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = to_gpu(data, self.is_gpu)
                output, _ = self.forward(data)

                nll_loss = self.nll_loss(data, output)
                total_nll_loss += nll_loss.item()

                adv_loss = self.adv_loss(data)
                total_adv_loss += adv_loss.item()

        total_loss = total_nll_loss + total_adv_loss
        len_data = len(test_loader.dataset)
        log_line("Epoch {} Test Loss : {:.4f} NLL Loss : {:.4f} Adv Loss : {:.4f}".format(
            epoch, total_loss / len_data, total_nll_loss / len_data,
                   total_adv_loss / len_data), log_file, is_print=True)

    def sample(self, epoch, sample_num, save_path):
        with torch.no_grad():
            random_noise = to_gpu(torch.randn(sample_num, self.nlatent), self.is_gpu)
            sample = self.decode(random_noise).cpu()
            save_image(sample.view(sample_num, 1, 28, 28),
                       save_path + 'epoch_' + str(epoch) + '.png')