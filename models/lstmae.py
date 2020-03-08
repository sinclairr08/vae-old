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

class LSTM_AE(nn.Module):
    def __init__(self, enc, dec, nlatent, ntokens, nembdim,
                 nlayers, nhidden, hidden_noise_r,
                 is_gpu):

        super(LSTM_AE, self).__init__()

        # Dimesions
        self.nlatent = nlatent
        self.ntokens = ntokens
        self.nembdim = nembdim
        self.nlayers = nlayers
        self.nhidden = nhidden
        self.hidden_noise_r = hidden_noise_r

        # Model infos
        self.enc = enc
        self.dec = dec

        self.word_dropout = 0.5

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

    def encode(self, input, lengths):
        embs = self.embedding_enc(input)
        packed_embs = pack_padded_sequence(
            input=embs, lengths=lengths, batch_first=True
        )

        packed_output, state = self.encoder(packed_embs)

        # mod : It is only possible when nlayers == 1 and Uni
        hidden = state[0][-1]

        # mod : Normalize to Gaussian
        # mod : argumentize
        hidden = hidden / torch.norm(hidden, p=2, dim=1, keepdim=True)

        self.is_hidden_noise = True

        if self.is_hidden_noise and self.hidden_noise_r > 0 :
            hidden_noise = torch.normal(mean=torch.zeros_like(hidden),
                                        std = self.hidden_noise_r)
            hidden  = hidden + to_gpu(Variable(hidden_noise), self.is_gpu)

        return hidden

    def noise_anneal(self, fac):
        self.hidden_noise_r *= fac

    def decode(self, hidden, batch_size, maxlen, input=None, lengths=None):

        # For the concatenation with embeddings
        hidden_expanded = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        # Replace the word as a <UNK> token
        # Steal from https://github.com/timbmg/Sentence-VAE/blob/master/model.py

        if self.word_dropout > 0:
            prob = to_gpu(torch.rand(input.size()), self.is_gpu)
            prob[(input.data - 1) * input.data == 0] = 1        # If the token is either <SOS> or <PAD>
            decoder_input = input.clone()
            decoder_input[prob < self.word_dropout] = 3         # 3 means <UNK> token index
            embs = self.embedding_dec(decoder_input)
        else:
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

    def loss(self, output, target, step):
        flattened_output = output.view(-1, self.ntokens)
        nll_loss = F.cross_entropy(flattened_output, target)

        return nll_loss


    def train_epoch(self, epoch, optimizer, train_loader, log_file, log_interval):
        self.train()
        total_nll_loss = 0
        total_len = 0

        for batch_idx, (source, target, lengths) in enumerate(train_loader):
            total_len += len(source)
            source = to_gpu(Variable(source), self.is_gpu)
            target = to_gpu(Variable(target), self.is_gpu)
            output = self.forward(source, lengths)
            optimizer.zero_grad()

            nll_loss = self.loss(output, target, batch_idx)
            nll_loss.backward()

            # mod : Argumentization of the clip
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            optimizer.step()

            total_nll_loss += nll_loss.item()

            if batch_idx % 100 == 0:
                self.noise_anneal(0.9995)

            if batch_idx % log_interval == 0 and batch_idx > 0:
                pass

        log_line("Epoch {} Train NLL Loss : {:.4f}".format(
            epoch, total_nll_loss / total_len),
            log_file, is_print=True)

        return total_nll_loss

    def test_epoch(self, epoch, test_loader, idx2word, log_file, save_path):
        self.eval()
        total_nll_loss = 0
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

                nll_loss = self.loss(output, target, batch_idx)
                total_nll_loss += nll_loss.item()

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

        log_line("Epoch {} Test NLL Loss : {:.4f}".format(
            epoch, total_nll_loss / total_len),
            log_file, is_print=True)

        avg_bleu1 = (avg_bleu1 / total_len) * 100
        avg_bleu2 = (avg_bleu2 / total_len) * 100
        avg_bleu3 = (avg_bleu3 / total_len) * 100
        avg_bleu4 = (avg_bleu4 / total_len) * 100
        avg_bleu5 = (avg_bleu5 / total_len) * 100

        avg_bleus = np.array([avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_bleu5])
        return avg_bleus

    # mod : Not yet
    def sample(self, epoch, sample_num, maxlen, idx2word, log_file, save_path, sample_method='sampling'):
        random_noise = to_gpu(torch.randn(sample_num, self.nhidden), self.is_gpu)

        start_symbols = to_gpu(Variable(torch.ones(sample_num, 1).long()), self.is_gpu)
        start_symbols.data.fill_(1)

        embs = self.embedding_dec(start_symbols)
        aug_embs = torch.cat([embs, random_noise.unsqueeze(1)], 2)

        all_token_indicies = []
        for i in range(maxlen):
            output, state = self.decoder(aug_embs)
            token_logits = self.hidden2token(output.squeeze(1))

            if sample_method == 'sampling':
                token_probs = F.softmax(token_logits, dim=-1)  # review that why it is -1
                token_indicies = torch.multinomial(token_probs, num_samples=1)

            elif sample_method == 'greedy':
                token_indicies = torch.argmax(token_logits, dim=1)

            else:
                raise NotImplementedError

            token_indicies = token_indicies.unsqueeze(1)
            all_token_indicies.append(token_indicies)

            # Use the previous output word as input
            embs = self.embedding_dec(token_indicies)
            if sample_method == 'sampling':
                embs = embs.squeeze(1)
            aug_embs = torch.cat([embs, random_noise.unsqueeze(1)], 2)

        cat_token_indicies = torch.cat(all_token_indicies, 1)

        if sample_method == 'sampling':
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

        selfbleu1 = SelfBleu(test_text=sampling_file, gram=1).get_score() * 100
        selfbleu2 = SelfBleu(test_text=sampling_file, gram=2).get_score() * 100
        selfbleu3 = SelfBleu(test_text=sampling_file, gram=3).get_score() * 100
        selfbleu4 = SelfBleu(test_text=sampling_file, gram=4).get_score() * 100
        selfbleu5 = SelfBleu(test_text=sampling_file, gram=5).get_score() * 100

        dist1 = UniqueGram(test_text=sampling_file, gram=1).get_score()
        dist2 = UniqueGram(test_text=sampling_file, gram=2).get_score()
        dist3 = UniqueGram(test_text=sampling_file, gram=3).get_score()
        dist4 = UniqueGram(test_text=sampling_file, gram=4).get_score()
        dist5 = UniqueGram(test_text=sampling_file, gram=5).get_score()

        selfbleus = np.array([selfbleu1, selfbleu2, selfbleu3, selfbleu4, selfbleu5])
        dists = np.array([dist1, dist2, dist3, dist4, dist5])
        return selfbleus, dists