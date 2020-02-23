import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # To calculate BLEU score

from metrics.uniquegram import UniqueGram
from metrics.selfbleu import SelfBleu

from utils import to_gpu, log_line

class LSTM_ARAE(nn.Module):
    def __init__(self, enc, dec, nlatent, ntokens, nembdim,
                 nlayers, nDhidden, nGhidden, nnoise,
                 is_gpu):

        super(LSTM_ARAE, self).__init__()

        # Dimesions
        self.nlatent = nlatent
        self.ntokens = ntokens
        self.nembdim = nembdim
        self.nlayers = nlayers
        self.nhidden = nlatent
        self.nDhidden = nDhidden
        self.nGhidden = nGhidden
        self.nnoise = nnoise

        # Model infos
        self.enc = enc
        self.dec = dec

        # Tools
        self.is_gpu = is_gpu

        # Consider more..

        # mod : dropout
        if self.enc == 'lstm':
            self.encoder = nn.LSTM(
                input_size=self.nembdim,
                hidden_size=self.nhidden,
                num_layers=self.nlayers,
                batch_first=True,
            )

        else:
            raise NotImplementedError

        # mod
        if self.dec == 'lstm':
            self.decoder = nn.LSTM(
                input_size=self.nembdim + self.nhidden,  # Decoder input size
                hidden_size=self.nhidden,
                num_layers=self.nlayers,
                batch_first=True,
            )

        else:
            raise NotImplementedError

        self.disc = nn.Sequential(
            nn.Linear(self.nlatent, self.nDhidden),
            nn.ReLU(),
            nn.Linear(self.nDhidden, 1),
            torch.nn.Sigmoid()
        )

        self.gen = nn.Sequential(
            nn.Linear(self.nnoise, self.nGhidden),
            nn.ReLU(),
            nn.Linear(self.nGhidden, self.nlatent),
            nn.ReLU(),
        )

        # optims
        self.eps = 1e-15

        # Layers
        self.embedding_enc = nn.Embedding(self.ntokens, nembdim)
        self.embedding_dec = nn.Embedding(self.ntokens, nembdim)

        self.hidden2token = nn.Linear(self.nhidden, self.ntokens)

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

        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.fill_(0)

    # Initialize the hidden state and cell state both
    # dep : Deprecatae since it is not used
    ''' 
    def init_hidden_cell(self, batch_size):
        hidden_state = to_gpu(Variable(torch.zeros(self.nlayers, batch_size, self.nhidden)), self.is_gpu)
        cell_state = to_gpu(Variable(torch.zeros(self.nlayers, batch_size, self.nhidden)), self.is_gpu)

        return (hidden_state, cell_state)
    '''

    def encode(self, input, lengths):
        embs = self.embedding_enc(input)
        packed_embs = pack_padded_sequence(
            input=embs, lengths=lengths, batch_first=True
        )

        packed_output, state = self.encoder(packed_embs)

        # mod : It is only possible when nlayers == 1 and Uni
        hidden = state[0][0]

        # mod : Normalize to Gaussian
        # mod : argumentize
        hidden = hidden / torch.norm(hidden, p=2, dim=1, keepdim=True)

        self.is_hidden_noise = True
        self.hidden_noise_r = 0.2

        if self.is_hidden_noise and self.hidden_noise_r > 0:
            hidden_noise = torch.normal(mean=torch.zeros_like(hidden),
                                        std=self.hidden_noise_r)
            hidden = hidden + to_gpu(Variable(hidden_noise), self.is_gpu)

        return hidden

    def decode(self, hidden, batch_size, maxlen, input=None, lengths=None):

        # For the concatenation with embeddings
        hidden_expanded = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        embs = self.embedding_dec(input)
        aug_embs = torch.cat([embs, hidden_expanded], 2)

        packed_embs = pack_padded_sequence(
            input=aug_embs, lengths=lengths, batch_first=True
        )

        packed_output, state = self.decoder(packed_embs)
        output_hidden, lengths = pad_packed_sequence(packed_output, batch_first=True)

        output = self.hidden2token(output_hidden.contiguous().view(-1, self.nhidden))
        output = output.view(batch_size, maxlen, self.ntokens)

        return output

    def forward(self, input, lengths, encode_only=False):

        batch_size, maxlen = input.size()

        hidden = self.encode(input, lengths)

        if encode_only:
            return hidden

        # mod : Register_hook
        output = self.decode(hidden, batch_size, maxlen, input, lengths)

        return output

    def nll_loss(self, output, target):
        flattened_output = output.view(-1, self.ntokens)
        loss = F.cross_entropy(flattened_output, target)

        return loss

    def disc_loss(self, latent_real, latent_fake):
        assert latent_real.shape == latent_fake.shape

        disc_real = self.disc(latent_real)
        disc_fake = self.disc(latent_fake)

        loss = -torch.mean(torch.log(disc_real + self.eps) + torch.log(1 - disc_fake + self.eps))

        return loss

    # deprecate
    def adv_loss_enc(self, input, lengths):
        latent_real = self.encode(input, lengths)
        disc_real = self.disc(latent_real)

        loss = -torch.mean(torch.log(1 - disc_real + self.eps))
        return loss

    # deprecate
    def adv_loss_gen(self, latent_fake):
        disc_fake = self.disc(latent_fake)

        loss = -torch.mean(torch.log(disc_fake + self.eps))
        return loss

    def train_epoch(self, epoch, train_loader,  optim_enc_nll, optim_enc_adv, optim_dec, optim_disc, optim_gen,
                    niters_gan, niters_ae, niters_gan_d, niters_gan_g, niters_gan_ae, log_file, log_interval):
        self.train()
        total_nll = 0
        total_err_adv = 0
        total_errD = 0
        total_errG = 0
        total_len = 0

        for batch_idx, (source, target, lengths) in enumerate(train_loader):
            total_len += len(source)
            source = to_gpu(Variable(source), self.is_gpu)
            target = to_gpu(Variable(target), self.is_gpu)

            one = to_gpu(torch.Tensor(len(source), 1).fill_(1), self.is_gpu)
            mone = one * -1

            # Phase 1 : Train Autoencoder
            for i in range(niters_ae):
                optim_enc_nll.zero_grad()
                optim_dec.zero_grad()
                output = self.forward(source, lengths)

                nll_loss = self.nll_loss(output, target)
                nll_loss.backward()
                total_nll += nll_loss.item()

                # mod : Argumentization of the clip
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optim_enc_nll.step()
                optim_dec.step()

            for j in range(niters_gan):
                # Phase 2 : Train Discriminator
                for k in range(niters_gan_d):
                    optim_disc.zero_grad()
                    '''
                    latent_real = self.encode(source, lengths)
                    random_noise = to_gpu(Variable(torch.randn(latent_real.shape[0], self.nnoise)),
                                          self.is_gpu)
                    latent_fake = self.gen(random_noise)

                    disc_loss = self.disc_loss(latent_real, latent_fake)
                    disc_loss.backward()
                    '''
                    latent_real = self.encode(source, lengths)
                    random_noise = to_gpu(Variable(torch.randn(latent_real.shape[0], self.nnoise)),
                                          self.is_gpu)
                    latent_fake = self.gen(random_noise)

                    errD_real = self.disc(latent_real.detach())
                    errD_fake = self.disc(latent_fake.detach())

                    errD_real.backward(one)
                    errD_fake.backward(mone)

                    errD = -(errD_real - errD_fake)
                    total_errD += torch.sum(errD).item()

                    optim_disc.step()

                # Phase 3 : Train encoder
                for k in range(niters_gan_ae):
                    optim_enc_adv.zero_grad()
                    '''
                    adv_loss_enc = self.adv_loss_enc(source, lengths)
                    adv_loss_enc.backward()
                    total_adv += adv_loss_enc.item()
                    '''
                    latent_real = self.encode(source, lengths)
                    err_adv = self.disc(latent_real)

                    err_adv.backward(mone)
                    total_err_adv += torch.sum(err_adv).item()
                    optim_enc_adv.step()

                # Phase 4 : Train generator using discriminator
                for k in range(niters_gan_g):
                    optim_gen.zero_grad()
                    '''
                    random_noise = to_gpu(Variable(torch.randn(latent_real.shape[0], self.nnoise)),
                                          self.is_gpu)
                    latent_fake = self.gen(random_noise)
                    adv_loss_gen = self.adv_loss_gen(latent_fake)
                    adv_loss_gen.backward()
                    '''
                    random_noise = to_gpu(Variable(torch.randn(latent_real.shape[0], self.nnoise)),
                                          self.is_gpu)
                    latent_fake = self.gen(random_noise)
                    errG = self.disc(latent_fake)
                    errG.backward(one)

                    total_errG += torch.sum(errG).item()
                    optim_gen.step()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                pass

        log_line("Epoch {} Train NLL Loss : {:.4f} Disc Loss : {:.4f} Adv Val : {:.4f} Gen Val : {:.4f}".format(
            epoch, total_nll / total_len, total_errD / total_len, total_err_adv / total_len, total_errG / total_len),
            log_file, is_print=True)

        return total_nll, total_errD, total_err_adv, total_errG

    def test_epoch(self, epoch, test_loader, idx2word, log_file, save_path):
        self.eval()
        total_nll = 0
        total_err_adv = 0
        total_len = 0

        avg_bleu1 = 0
        avg_bleu2 = 0
        avg_bleu3 = 0
        avg_bleu4 = 0
        avg_bleu5 = 0
        cc = SmoothingFunction()

        epoch_ae_generated_file = os.path.join(save_path, "epoch_" + str(epoch) + "_ae_generation.txt")

        with torch.no_grad():
            for batch_idx, (source, target, lengths) in enumerate(test_loader):
                total_len += len(source)
                source = to_gpu(Variable(source), self.is_gpu)
                target = to_gpu(Variable(target), self.is_gpu)

                output = self.forward(source, lengths)

                nll_loss = self.nll_loss(output, target)
                total_nll += nll_loss.item()
                '''
                adv_loss = self.adv_loss_enc(source, lengths)
                total_adv += adv_loss.item()
                '''

                latent_real = self.encode(source, lengths)
                err_adv = self.disc(latent_real)

                total_err_adv += torch.sum(err_adv).item()

                with open(epoch_ae_generated_file, "a") as f:
                    max_values, max_indices = torch.max(output, 2)
                    max_indices = max_indices.view(output.size(0), -1).data.cpu().numpy()
                    target = target.view(output.size(0), -1).data.cpu().numpy()
                    for t, idx in zip(target, max_indices):
                        # real sentence
                        chars = " ".join([idx2word[x] for x in t])
                        f.write(chars)
                        f.write("\n")

                        reference = chars.split()

                        # autoencoder output sentence
                        chars = " ".join([idx2word[x] for x in idx])
                        f.write(chars)
                        f.write("\n\n")
                        candidate = chars.split()

                        gram1 = sentence_bleu([reference], candidate, smoothing_function=cc.method1,
                                              weights=(1, 0, 0, 0, 0), auto_reweigh=True)
                        gram2 = sentence_bleu([reference], candidate, smoothing_function=cc.method1,
                                              weights=(1 / 2, 1 / 2, 0, 0, 0), auto_reweigh=True)
                        gram3 = sentence_bleu([reference], candidate, smoothing_function=cc.method1,
                                              weights=(1 / 3, 1 / 3, 1 / 3, 0, 0), auto_reweigh=True)
                        gram4 = sentence_bleu([reference], candidate, smoothing_function=cc.method1,
                                              weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4, 0), auto_reweigh=True)
                        gram5 = sentence_bleu([reference], candidate, smoothing_function=cc.method1,
                                              weights=(1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5), auto_reweigh=True)

                        avg_bleu1 += gram1
                        avg_bleu2 += gram2
                        avg_bleu3 += gram3
                        avg_bleu4 += gram4
                        avg_bleu5 += gram5

        log_line("Epoch {} Test NLL Loss : {:.4f} Adv Val : {:.4f}".format(
            epoch, total_nll / total_len, total_err_adv / total_len),
            log_file, is_print=True)

        log_line("BLEU-1: {:.3f}".format((avg_bleu1 / total_len) * 100), log_file,is_print=True)
        log_line("BLEU-2: {:.3f}".format((avg_bleu2 / total_len) * 100), log_file, is_print=True)
        log_line("BLEU-3: {:.3f}".format((avg_bleu3 / total_len) * 100), log_file, is_print=True)
        log_line("BLEU-4: {:.3f}".format((avg_bleu4 / total_len) * 100), log_file, is_print=True)
        log_line("BLEU-5: {:.3f}".format((avg_bleu5 / total_len) * 100), log_file, is_print=True)

    # mod : Not yet
    def sample(self, epoch, sample_num, maxlen, idx2word, log_file, save_path, sample_method='sampling'):
        random_noise = to_gpu(torch.randn(sample_num, self.nnoise), self.is_gpu)
        latent_synth = self.gen(random_noise)

        start_symbols = to_gpu(Variable(torch.ones(sample_num, 1).long()), self.is_gpu)
        start_symbols.data.fill_(1)

        embs = self.embedding_dec(start_symbols)
        aug_embs = torch.cat([embs, latent_synth.unsqueeze(1)], 2)

        all_token_indicies = []
        for i in range(maxlen):
            output, state = self.decoder(aug_embs)
            token_logits = self.hidden2token(output.squeeze(1))

            if sample_method == 'sampling':
                # print(token_logits)
                token_probs = F.softmax(token_logits, dim=-1)  # review that why it is -1
                # print(token_probs)

                # mod : Error that negative prob
                try:
                    token_indicies = torch.multinomial(token_probs, num_samples=1)
                except:
                    print(token_probs)
                    exit(-1)

            elif sample_method == 'greedy':
                token_indicies = torch.argmax(token_logits, dim=1)

            else:
                raise NotImplementedError

            token_indicies = token_indicies.unsqueeze(1)
            all_token_indicies.append(token_indicies)

            # Use the previous output word as input
            embs = self.embedding_dec(token_indicies)
            embs = embs.squeeze(1)
            aug_embs = torch.cat([embs, latent_synth.unsqueeze(1)], 2)

        cat_token_indicies = torch.cat(all_token_indicies, 1)

        # print(cat_token_indicies.shape)
        cat_token_indicies = cat_token_indicies.squeeze(2)
        cat_token_indicies = cat_token_indicies.data.cpu().numpy()

        sampling_file = os.path.join(save_path, 'epoch_' + str(epoch) + "_sampling.txt")
        sentences = []
        for idx in cat_token_indicies:
            words = [idx2word[x] for x in idx]

            sentence_list = []

            for word in words:
                if word != '<eos>':
                    sentence_list.append(word)
                else:
                    break

            sentence = " ".join(sentence_list)

            log_line(sentence, sampling_file, is_print=False)
            sentences.append(sentence)

        selfbleu1 = SelfBleu(test_text=sampling_file, gram=1).get_score()
        selfbleu2 = SelfBleu(test_text=sampling_file, gram=2).get_score()
        selfbleu3 = SelfBleu(test_text=sampling_file, gram=3).get_score()
        selfbleu4 = SelfBleu(test_text=sampling_file, gram=4).get_score()
        selfbleu5 = SelfBleu(test_text=sampling_file, gram=5).get_score()

        dist1 = UniqueGram(test_text=sampling_file, gram=1).get_score()
        dist2 = UniqueGram(test_text=sampling_file, gram=2).get_score()
        dist3 = UniqueGram(test_text=sampling_file, gram=3).get_score()
        dist4 = UniqueGram(test_text=sampling_file, gram=4).get_score()
        dist5 = UniqueGram(test_text=sampling_file, gram=5).get_score()

        log_line('Self-BLEU 1: {:.3f}'.format(selfbleu1 * 100), log_file, is_print=True)
        log_line('Self-BLEU 2: {:.3f}'.format(selfbleu2 * 100), log_file, is_print=True)
        log_line('Self-BLEU 3: {:.3f}'.format(selfbleu3 * 100), log_file, is_print=True)
        log_line('Self-BLEU 4: {:.3f}'.format(selfbleu4 * 100), log_file, is_print=True)
        log_line('Self-BLEU 5: {:.3f}'.format(selfbleu5 * 100), log_file, is_print=True)

        log_line('Dist 1: {:.3f}'.format(dist1), log_file, is_print=True)
        log_line('Dist 2: {:.3f}'.format(dist2), log_file, is_print=True)
        log_line('Dist 3: {:.3f}'.format(dist3), log_file, is_print=True)
        log_line('Dist 4: {:.3f}'.format(dist4), log_file, is_print=True)
        log_line('Dist 5: {:.3f}'.format(dist5), log_file, is_print=True)

        return