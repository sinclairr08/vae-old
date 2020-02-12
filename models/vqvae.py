import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image

from utils import to_gpu, log_line

class VQEmbedding(nn.Module):
    def __init__(self, nemb, nembdim, commit_cost, is_gpu):
        super(VQEmbedding, self).__init__()

        self.nemb = nemb
        self.nembdim = nembdim
        self.commit_cost = commit_cost
        self.is_gpu = is_gpu

        self.embedding = nn.Embedding(self.nemb, self.nembdim)
        self.embedding.weight.data.uniform_(-1/self.nemb, 1/self.nemb)

    # Quantize the input vector to the nearset one
    def forward(self, input):
        # Set the Batch * Colors * Height * Width -> Batch * Height * Width * Colors

        input_shape = input.shape
        flattened_input = input.view(-1, self.nembdim)  # size : N * DIM

        # Calculate the distances for each "segment" of embeddings
        # Note : (x - y)^2 == x^2 + y^2 -2(x^T) y
        distances = (torch.sum(flattened_input ** 2, dim = 1, keepdim=True)
                   + torch.sum(self.embedding.weight**2, dim = 1)
                   - torch.matmul(flattened_input, self.embedding.weight.t()))

        indices = torch.argmin(distances, dim = 1).unsqueeze(1) # Formed as [[1], [3], ...]
        encodings = to_gpu(torch.zeros(indices.shape[0], self.nemb), self.is_gpu) # Formed as [[0,0,...,0], ...]
        encodings.scatter_(1, indices, 1)            # Formed as [[0,1,0,0], [0,0,0,1], ...] N * dim
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)


        # Codebook Loss
        code_loss = F.mse_loss(quantized, input.detach())
        # Commitment loss
        comit_loss = F.mse_loss(quantized.detach(), input)

        loss = code_loss + self.commit_cost * comit_loss # Commitment cost
        quantized = input + (quantized - input).detach()

        return quantized.contiguous(), loss

class VQ_VAE(nn.Module):
    def __init__(self, nlatent, ninput, nhidden,
                 nemb, nembdim, commit_cost, is_gpu):

        super(VQ_VAE, self).__init__()
        # self init area
        self.is_gpu = is_gpu
        self.nlatent = nlatent
        self.ninput = ninput
        self.nhidden = nhidden

        self.codebook = VQEmbedding(nemb, nembdim, commit_cost, self.is_gpu)

        # mod : Modlarize within the class
        # self.enc = None
        # self.dec = None

        self.encoder = nn.Sequential(
            nn.Linear(self.ninput, self.nhidden),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(self.nhidden, self.nhidden),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(self.nhidden, self.nlatent),
        )

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


    def encode(self, input):
        encoder_output = self.encoder(input.view(-1, self.ninput))
        latent, loss = self.codebook(encoder_output)
        return latent, loss

    # mod : Modularization
    def decode(self, latent):
        output = self.decoder(latent)
        return output

    # Returns the input to the output, mu, and log var
    def forward(self, input):
        latent, loss = self.encode(input)
        output = self.decode(latent)
        return output, loss

    def loss(self, input, output):
        nll_loss = F.mse_loss(input.view(-1, self.ninput), output)
        return nll_loss


    def train_epoch(self, epoch, optimizer, train_loader, log_file):
        self.train()
        total_loss = 0
        total_vq_loss = 0
        total_nll_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = to_gpu(data, self.is_gpu)
            output, vq_loss = self.forward(data)
            optimizer.zero_grad()

            nll_loss = self.loss(data, output)
            loss = nll_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nll_loss += nll_loss.item()
            total_vq_loss += vq_loss.item()

        total_len = len(train_loader.dataset)
        log_line("Epoch {} Train Loss : {:.4f} NLL Loss : {:.4f} VQ Loss : {:.4f}".format(
            epoch, total_loss / total_len, total_nll_loss / total_len, total_vq_loss / total_len),
            log_file, is_print=True, is_line=True)

    def test_epoch(self, epoch, test_loader, log_file):
        self.eval()
        total_loss = 0
        total_nll_loss = 0
        total_vq_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = to_gpu(data, self.is_gpu)
                output, vq_loss = self.forward(data)
                nll_loss = self.loss(data, output)

                loss = nll_loss + vq_loss

                total_nll_loss += nll_loss.item()
                total_vq_loss += vq_loss.item()
                total_loss += loss.item()

        total_len = len(test_loader.dataset)
        log_line("Epoch {} Test Loss : {:.4f} NLL Loss : {:.4f} VQ Loss : {:.4f}".format(
            epoch, total_loss / total_len, total_nll_loss / total_len, total_vq_loss / total_len),
            log_file, is_print=True)

    def sample(self, epoch, sample_num, save_path):
        with torch.no_grad():
            random_noise = to_gpu(torch.randn(sample_num, self.nlatent), self.is_gpu)
            sample = self.decode(random_noise).cpu()
            save_image(sample.view(sample_num, 1, 28, 28),
                       save_path + 'epoch_' + str(epoch) + '.png')
