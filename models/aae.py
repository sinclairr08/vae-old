import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

from utils import to_gpu, log_line

from discriminator import MLP_D

class AAE(nn.Module):
    def __init__(self, nlatent, ninput, nhidden, D_arch, is_gpu):

        super(AAE, self).__init__()
        # self init area
        self.is_gpu = is_gpu
        self.nlatent = nlatent
        self.ninput = ninput
        self.nhidden = nhidden
        self.D_arch = D_arch

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
        self.disc = MLP_D(ninput=self.nlatent, noutput=1, layers=self.D_arch, activation=nn.LeakyReLU(0.2),
                          is_gpu = is_gpu)

        '''
        nn.Sequential(
            nn.Linear(self.nlatent, self.nDhidden),
            nn.ReLU(),
            nn.Linear(self.nDhidden, self.nDhidden),
            nn.ReLU(),
            nn.Linear(self.nDhidden, 1),
        )
        '''

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

    # deprecate
    def disc_loss(self, latent_real, latent_fake):
        assert latent_real.shape == latent_fake.shape

        disc_real = self.disc(latent_real)
        disc_fake = self.disc(latent_fake)

        loss = -torch.mean(torch.log(disc_real + self.eps) + torch.log(1 - disc_fake + self.eps))

        return loss

    # deprecate
    def adv_loss(self, input):
        latent_fake = self.encode(input)
        disc_fake = self.disc(latent_fake)

        loss = -torch.mean(torch.log(disc_fake + self.eps))
        return loss

    def train_epoch(self, epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc,
                    niters_gan, niters_ae, niters_gan_d, niters_gan_ae, log_file):
        self.train()

        total_nll = 0
        total_err_adv = 0
        total_errD = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = to_gpu(data, self.is_gpu)

            one = to_gpu(torch.Tensor(len(data), 1).fill_(1), self.is_gpu)
            mone = one * -1

            # Phase 1 : Train Autoencoder
            for i in range(niters_ae):
                optim_enc_nll.zero_grad()
                optim_dec.zero_grad()
                output, _ = self.forward(data)

                nll_loss = self.nll_loss(data, output)  # Need to check the names
                nll_loss.backward()
                total_nll += nll_loss.item()
                optim_enc_nll.step()
                optim_dec.step()

            for j in range(niters_gan):

                # Phase 2 : Train Discriminator
                for k in range(niters_gan_d):
                    optim_disc.zero_grad()
                    '''
                    latent_fake = self.encode(data)
                    latent_real = to_gpu(Variable(torch.randn_like(latent_fake)), self.is_gpu)

                    disc_loss = self.disc_loss(latent_real, latent_fake)
                    disc_loss.backward()
                    '''
                    latent_fake = self.encode(data)
                    latent_real = to_gpu(Variable(torch.randn_like(latent_fake)), self.is_gpu)

                    errD_fake = self.disc(latent_fake.detach())
                    errD_real = self.disc(latent_real.detach())

                    errD_fake.backward(mone)
                    errD_real.backward(one)

                    errD = -(errD_real - errD_fake)
                    total_errD += torch.sum(errD).item()

                    optim_disc.step()

                # Phase 3 : Train encoder
                for k in range(niters_gan_ae):
                    optim_enc_adv.zero_grad()
                    '''
                    adv_loss = self.adv_loss(data)
                    adv_loss.backward()
                    total_adv += adv_loss.item()
                    '''
                    latent_fake = self.encode(data)
                    err_adv = self.disc(latent_fake)

                    err_adv.backward(one)
                    total_err_adv += torch.sum(err_adv).item()
                    optim_enc_adv.step()

        total_len = len(train_loader.dataset)
        log_line("Epoch {} Train NLL Loss : {:.4f} Disc Loss : {:.4f} Adv Val : {:.4f}".format(
            epoch, total_nll / total_len, total_errD / total_len, total_err_adv / total_len), log_file, is_print=True)

    def test_epoch(self, epoch, test_loader, log_file):
        self.eval()
        total_nll = 0
        total_err_adv = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = to_gpu(data, self.is_gpu)
                output, _ = self.forward(data)

                nll_loss = self.nll_loss(data, output)
                total_nll += nll_loss.item()

                latent_fake = self.encode(data)
                err_adv = self.disc(latent_fake)

                total_err_adv += torch.sum(err_adv).item()

        total_len = len(test_loader.dataset)
        log_line("Epoch {} Test NLL Loss : {:.4f} Adv Val : {:.4f}".format(
            epoch, total_nll / total_len, total_err_adv / total_len), log_file, is_print=True)

    def sample(self, epoch, sample_num, save_path):
        with torch.no_grad():
            random_noise = to_gpu(torch.randn(sample_num, self.nlatent), self.is_gpu)
            sample = self.decode(random_noise).cpu()
            save_image(sample.view(sample_num, 1, 28, 28),
                       save_path + 'epoch_' + str(epoch) + '.png')