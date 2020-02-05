import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.utils import save_image

from utils import to_gpu, log_line

# Reference : https://github.com/pytorch/examples/tree/master/vae
# Have to add at the readme.md

class VAE(nn.Module):
    def __init__(self, nlatent, ninput, nhidden,  is_gpu):

        super(VAE, self).__init__()
        # self init area
        self.is_gpu = is_gpu
        self.nlatent = nlatent
        self.ninput = ninput #com2
        self.nhidden = nhidden #com2

        # mod : Modlarize within the class
        #self.enc = None
        #self.dec = None

        # Temp code
        self.enc1 = nn.Linear(self.ninput, self.nhidden)
        self.enc21 = nn.Linear(self.nhidden, self.nlatent)
        self.enc22 = nn.Linear(self.nhidden, self.nlatent)
        self.dec1 = nn.Linear(self.nlatent, self.nhidden)
        self.dec2 = nn.Linear(self.nhidden, self.ninput)

    # mod : Modularization
    # mod : layer of mu, logvar
    def encode(self, input):
        """
        if self.enc is None:
            raise NotImplementedError
        return self.enc(input)
        """
        h1 = F.relu(self.enc1(input))
        return self.enc21(h1), self.enc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = to_gpu(torch.rand_like(self.std), self.is_gpu) # epsilon for reparametrization
        return mu + eps*std

    # mod : Modularization
    def decode(self, latent):
        """
        if self.dec is None:
            raise NotImplementedError
        return self.dec(latent)
        """
        h2 = F.relu(self.dec1(latent))
        return torch.sigmoid(self.dec2(h2))

    # Returns the input to the output, mu, and log var
    def forward(self, input):
        mu, logvar = self.encode(input.view(-1, self.ninput))
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
        optimizer.zero_grad()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = to_gpu(data, self.is_gpu)
            output, mu, logvar = self.forward(data)

            loss = self.loss(data, output, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        log_line("Epoch {} Train Loss : {:.4f}".format(
            epoch, total_loss / len(train_loader.dataset)),log_file,  is_print=True, is_line = False)

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
            total_loss / len(test_loader.dataset)), log_file,  is_print=True)

    def sample(self, epoch, sample_num, save_path):
        with torch.no_grad():
            latent_sample = to_gpu(torch.randn(sample_num, self.nlatent), self.is_gpu)
            sample = self.decode(latent_sample).cpu()
            save_image(sample.view(sample_num, 1, 28, 28),
                       save_path + 'epoch_' + str(epoch) + '.png')


class LSTM_VAE(nn.Module):
    def __init__(self, enc, dec, nlatent, ntokens, nemb,
                 nlayers, nhidden,
                 is_gpu):

        super(LSTM_VAE, self).__init__()

        # Dimesions
        self.nlatent = nlatent
        self.ntokens = ntokens
        self.nemb = nemb
        self.nlayers = nlayers
        self.nhidden = nhidden

        # Model infos
        self.enc = enc
        self.dec = dec

        # Tools
        self.is_gpu = is_gpu

        # Consider more..

        # mod : dropout
        if self.enc == 'lstm':
            self.encoder = nn.LSTM(
                input_size= self.nemb,
                hidden_size= self.nhidden,
                num_layers= self.nlayers,
                batch_first=True,
            )

        else:
            raise NotImplementedError

        # mod
        if self.dec == 'lstm':
            self.decoder = nn.LSTM(
                input_size=self.nemb + self.nhidden, # Decoder input size
                hidden_size= self.nhidden,
                num_layers=self.nlayers,
                batch_first=True,
            )

        else:
            raise NotImplementedError

        # Layers
        self.embedding_enc = nn.Embedding(self.ntokens, nemb)
        self.embedding_dec = nn.Embedding(self.ntokens, nemb)

        self.hidden2mean = nn.Linear(self.nhidden, self.nlatent)
        self.hidden2logvar = nn.Linear(self.nhidden, self.nlatent)
        self.latent2hidden = nn.Linear(self.nlatent, self.nhidden)
        self.hidden2token =  nn.Linear(self.nhidden, self.ntokens)

        self.init_weights()

    # Initialize the weights of LSTM and VAE
    def init_weights(self):
        initrange = 0.1

        self.embedding_enc.weight.data.uniform_(-initrange, initrange)
        self.embedding_dec.weight.data.uniform_(-initrange, initrange)

        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        self.hidden2mean.weight.data.uniform_(-initrange, initrange)
        self.hidden2mean.bias.data.fill_(0)
        self.hidden2logvar.weight.data.uniform_(-initrange, initrange)
        self.hidden2logvar.bias.data.fill_(0)
        self.latent2hidden.weight.data.uniform_(-initrange, initrange)
        self.latent2hidden.bias.data.fill_(0)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.fill_(0)

    # Initialize the hidden state and cell state both
    def init_hidden_cell(self, batch_size):
        hidden_state = to_gpu(Variable(torch.zeros(self.nlayers, batch_size, self.nhidden)), self.is_gpu)
        cell_state = to_gpu(Variable(torch.zeros(self.nlayers, batch_size, self.nhidden)), self.is_gpu)

        return (hidden_state, cell_state)

    def encode(self, input, lengths):
        embs = self.embedding_enc(input)
        packed_embs = pack_padded_sequence(
            input=embs, lengths = lengths, batch_first = True
        )

        packed_output, state = self.encoder(packed_embs)

        # mod : It is only possible when nlayers == 1 and Uni
        hidden = state[0][0]

        # mod : Normalize to Gaussian
        return hidden

    def reparameterize(self, hidden):
        self.mu = self.hidden2mean(hidden)
        self.logvar = self.hidden2logvar(hidden)
        self.std = torch.exp(0.5*self.logvar)

        eps = to_gpu(torch.rand_like(self.std), self.is_gpu) # epsilon for reparametrization
        latent = self.mu + self.std * eps

        return latent

    def decode(self, hidden, batch_size, maxlen, input = None, lengths = None):

        # For the concatenation with embeddings
        hidden_expanded = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        # mod : Decoder word dropout : input에 장난질 치기
        # mod : -> input이 일정 확률보다 낮으면 UNK로

        embs = self.embedding_dec(input)
        aug_embs = torch.cat([embs, hidden_expanded], 2)

        packed_embs = pack_padded_sequence(
            input=aug_embs, lengths=lengths, batch_first=True
        )

        # mod : NEED INITIALIZED STATE
        """
        state = self.init_hidden(batch_size)
        packed_output, state =self.decoder(packed_embs, state)
        
        """

        packed_output, state = self.decoder(packed_embs)
        output_hidden, lengths = pad_packed_sequence(packed_output, batch_first = True)

        output = self.hidden2token(output_hidden.contiguous().view(-1, self.nhidden))
        output = output.view(batch_size, maxlen, self.ntokens)

        return output


    def forward(self, input, lengths, encode_only = False):

        batch_size, maxlen = input.size()

        hidden_enc = self.encode(input, lengths)
        latent = self.reparameterize(hidden_enc)

        hidden_dec = self.latent2hidden(latent)

        if encode_only:
            return hidden_dec

        # mod : Register_hook
        output = self.decode(hidden_dec, batch_size, maxlen, input, lengths)

        return output

    def loss(self, output, target, step):
        flattened_output = output.view(-1, self.ntokens)
        nll_loss = F.cross_entropy(flattened_output, target)
        kl_loss = -0.5 * torch.sum(1 + self.logvar - self.mu ** 2 - self.logvar.exp())
        kl_loss /= len(output)

        kl_weight = self.kl_anneal_fn(step)

        return nll_loss + kl_weight * kl_loss, nll_loss, kl_loss

    # mode : ARGUMENTIZATION
    def kl_anneal_fn(self, step):
        k = 0.0025
        x0 = 2500
        return float(1 / (1 + np.exp(-k * (step - x0))))

    def train_epoch(self, epoch, optimizer, train_loader, log_file, log_interval):
        self.train()
        optimizer.zero_grad()
        total_loss = 0
        total_nll = 0
        total_kl = 0
        total_len = 0

        for batch_idx, (source, target, lengths) in enumerate(train_loader):
            total_len += len(source)
            source = to_gpu(Variable(source), self.is_gpu)
            target = to_gpu(Variable(target), self.is_gpu)
            output = self.forward(source, lengths)

            loss, nll_loss, kl_loss = self.loss(output, target, batch_idx)
            loss.backward()

            # mod : clip norm

            optimizer.step()

            total_loss += loss.item()
            total_nll += nll_loss.item()
            total_kl += kl_loss.item()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                pass

        log_line("Epoch {} Train Loss : {:.4f} NLL Loss : {:.4f} KL Loss : {:.4f}".format(
            epoch, total_loss / total_len, total_nll / total_len, total_kl / total_len), log_file, is_print=True)

        return total_loss, total_nll, total_kl

    def test_epoch(self, epoch, test_loader, idx2word, log_file, save_path):
        self.eval()
        total_loss = 0
        total_nll = 0
        total_kl = 0
        total_len = 0

        epoch_ae_generated_file = os.path.join(save_path, "epoch_" + str(epoch) + "_ae_generation.txt")

        with torch.no_grad():
            for batch_idx, (source, target, lengths) in enumerate(test_loader):
                total_len += len(source)
                source = to_gpu(Variable(source), self.is_gpu)
                target = to_gpu(Variable(target), self.is_gpu)

                output = self.forward(source, lengths)

                loss, nll_loss, kl_loss = self.loss(output, target, batch_idx)
                total_loss += loss.item()
                total_nll += nll_loss.item()
                total_kl += kl_loss.item()


                with open(epoch_ae_generated_file, "a") as f:
                    max_values, max_indices = torch.max(output, 2)
                    max_indices = max_indices.view(output.size(0), -1).data.cpu().numpy()
                    target = target.view(output.size(0), -1).data.cpu().numpy()
                    for t, idx in zip(target, max_indices):
                        # real sentence
                        chars = " ".join([idx2word[x] for x in t])
                        f.write(chars)
                        f.write("\n")

                        # autoencoder output sentence
                        chars = " ".join([idx2word[x] for x in idx])
                        f.write(chars)
                        f.write("\n\n")


        log_line("Epoch {} Test Loss : {:.4f} NLL Loss : {:.4f} KL Loss : {:.4f}".format(
            epoch, total_loss / total_len, total_nll / total_len, total_kl / total_len),
            log_file, is_print=True)

    # mod : Not yet
    def sample(self, epoch, sample_num, save_path):
        start_symbols = to_gpu(Variable(torch.ones(sample_num, 1).long()), self.is_gpu)
        start_symbols.data.fill_(1)

        embs = self.embedding_dec(start_symbols)
        #aug_embs = torch.cat([embs, latent_expanded], 2)

        pass















