import argparse
import os

import random
import numpy as np
import torch
import torch.utils.data


from torch import optim
from torchvision import datasets, transforms

from models.vae import VAE
from models.aae import AAE
from models.arae import ARAE
from models.lstmvae import LSTM_VAE
from models.lstmaae import LSTM_AAE
from models.lstmarae import LSTM_ARAE
from models.vqvae import VQ_VAE
from models.lstmvqvae import LSTM_VQ_VAE
from models.lstmvqvae2 import LSTM_VQ_VAE2
from models.lstmae import LSTM_AE

from models.lm import train_lm

from utils import to_gpu, batchify, lstm_scores, log_lstm_scores
from preprocess import Corpus
from config import config_args

def main(args):
    # dataset loading
    if args.dataset == 'mnist':
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # mod : Is it really need?

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )

    elif args.dataset == 'snli' or args.dataset == 'bc':
        corpus = Corpus(args.data_path,
                        maxlen=args.maxlen,
                        vocab_size=args.nvocab,
                        lowercase=args.lowercase)
        ntokens = len(corpus.dictionary.word2idx)

        train_loader = batchify(corpus.train, args.batch_size, shuffle=True, is_gpu=args.cuda)
        test_loader = batchify(corpus.test, args.batch_size, shuffle=False, is_gpu=args.cuda)

    else:
        raise NotImplementedError

    # Case 1 : VAE
    if args.model == 'vae':
        Model = VAE(nlatent=args.nlatent,
                    ninput=args.ninput,
                    nhidden=args.nhidden,
                    is_gpu = args.cuda)
        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr_ae)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file)
            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)  ## MODIFICATION

    # Case 2 : AAE
    elif args.model == 'aae':
        Model = AAE(nlatent=args.nlatent,
                    ninput=args.ninput,
                    nhidden=args.nhidden,
                    D_arch=args.D_arch,
                    is_gpu = args.cuda)
        Model = to_gpu(Model, args.cuda)

        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)

        if args.niters_gan_schedule is not None:
            gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
        else:
            gan_schedule = []

        niters_gan = 1

        for epoch in range(1, args.epochs + 1):
            if epoch in gan_schedule:
                niters_gan += 1
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc,
                              niters_gan, args.niters_ae, args.niters_gan_d, args.niters_gan_ae,
                              args.log_file)
            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)

    # Case 3 : ARAE
    elif args.model == 'arae':
        Model = ARAE(nlatent=args.nlatent,
                    ninput=args.ninput,
                    nhidden=args.nhidden,
                    D_arch=args.D_arch,
                    G_arch=args.G_arch,
                    nnoise= args.nnoise,
                    is_gpu = args.cuda)
        Model = to_gpu(Model, args.cuda)

        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)
        optim_gen = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_G)

        if args.niters_gan_schedule is not None:
            gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
        else:
            gan_schedule = []

        niters_gan = 1

        for epoch in range(1, args.epochs + 1):
            if epoch in gan_schedule:
                niters_gan += 1
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc, optim_gen,
                              niters_gan, args.niters_ae, args.niters_gan_d, args.niters_gan_g, args.niters_gan_ae,
                              args.log_file)

            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)

    # Case 4 : VQ VAE (Need more automization)
    elif args.model == 'vqvae':
        Model = VQ_VAE(nlatent=args.nlatent,
                    ninput=args.ninput,
                    nhidden=args.nhidden,
                    nemb=args.nemb,
                    nembdim = args.nembdim,
                    commit_cost=args.commit_cost,
                    is_gpu = args.cuda)
        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr_ae)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file)
            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)  ## MODIFICATION

    # Case 5 : LSTM-VAE
    elif args.model == 'lstmvae':        # mod : bc or other datasets
        Model = LSTM_VAE(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nembdim = args.nembdim,
                         nlayers= args.nlayers,
                         nhidden= args.nhidden,
                         hidden_noise_r = args.noise_r,
                         word_dropout = args.word_dropout,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr_ae)

        bleus = np.array([])
        selfbleus = np.array([])
        dists = np.array([])
        rev_ppls = np.array([])
        for_ppls = np.array([])

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file, args.log_interval)
            ep_bleus = Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            ep_selfbleus, ep_dists = Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         log_file = args.log_file, save_path=args.save_path,
                                                   sample_method = 'greedy')
                                                   # sample_method='sampling')

            rev_ppl, for_ppl = train_lm(data_path=args.data_path, save_path=args.save_path, maxlen=args.maxlen,
                                        sample_method='greedy',
                                        dict=corpus.dictionary, N=5, epoch=epoch, model=Model, log_file=args.log_file)

            bleus, selfbleus, dists, rev_ppls, for_ppls = lstm_scores(ep_bleus, ep_selfbleus, ep_dists, rev_ppl,
                                                                      for_ppl,
                                                                      bleus, selfbleus, dists, rev_ppls, for_ppls)

        log_lstm_scores(bleus, selfbleus, dists, rev_ppls, for_ppls, args.log_file)

    # Case 6 : LSTM-AAE
    elif args.model == 'lstmaae':        # mod : bc or other datasets
        Model = LSTM_AAE(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nembdim = args.nembdim,
                         nlayers= args.nlayers,
                         D_arch=args.D_arch,
                         hidden_noise_r=args.noise_r,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)

        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)

        if args.niters_gan_schedule is not None:
            gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
        else:
            gan_schedule = []

        niters_gan = 1

        bleus = np.array([])
        selfbleus = np.array([])
        dists = np.array([])
        rev_ppls = np.array([])
        for_ppls = np.array([])

        for epoch in range(1, args.epochs + 1):
            if epoch in gan_schedule:
                niters_gan += 1
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc,
                              niters_gan, args.niters_ae, args.niters_gan_d, args.niters_gan_ae, args.gp,
                              args.log_file, args.log_interval)
            ep_bleus= Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            ep_selfbleus, ep_dists = Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         log_file = args.log_file, save_path=args.save_path,
                                                   sample_method='greedy')
                                                   #sample_method='sampling')

            rev_ppl, for_ppl = train_lm(data_path=args.data_path, save_path=args.save_path, maxlen=args.maxlen,
                                        sample_method='greedy',
                                        dict=corpus.dictionary, N=5, epoch=epoch, model=Model, log_file=args.log_file)

            bleus, selfbleus, dists, rev_ppls, for_ppls = lstm_scores(ep_bleus, ep_selfbleus, ep_dists, rev_ppl,
                                                                      for_ppl,
                                                                      bleus, selfbleus, dists, rev_ppls, for_ppls)

        log_lstm_scores(bleus, selfbleus, dists, rev_ppls, for_ppls, args.log_file)

    # Case 7 : LSTM-ARAE
    elif args.model == 'lstmarae':        # mod : bc or other datasets
        Model = LSTM_ARAE(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nembdim = args.nembdim,
                         nlayers= args.nlayers,
                         D_arch=args.D_arch,
                         G_arch=args.G_arch,
                         nnoise= args.nnoise,
                         hidden_noise_r=args.noise_r,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)

        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)
        optim_gen = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_G)

        if args.niters_gan_schedule is not None:
            gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
        else:
            gan_schedule = []

        niters_gan = 1

        bleus = np.array([])
        selfbleus = np.array([])
        dists = np.array([])
        rev_ppls = np.array([])
        for_ppls = np.array([])

        for epoch in range(1, args.epochs + 1):
            if epoch in gan_schedule:
                niters_gan += 1
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc, optim_gen,
                              niters_gan, args.niters_ae, args.niters_gan_d, args.niters_gan_g, args.niters_gan_ae,
                              args.gp, args.log_file, args.log_interval)
            ep_bleus = Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            ep_selfbleus, ep_dists = Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         log_file = args.log_file, save_path=args.save_path,
                                                   sample_method='greedy')
                                                   #sample_method='sampling')
            rev_ppl, for_ppl = train_lm(data_path=args.data_path, save_path=args.save_path, maxlen=args.maxlen, sample_method='greedy',
                     dict=corpus.dictionary, N=5, epoch=epoch, model= Model, log_file=args.log_file)

            bleus, selfbleus, dists, rev_ppls, for_ppls = lstm_scores(ep_bleus, ep_selfbleus, ep_dists, rev_ppl, for_ppl,
                                                  bleus, selfbleus, dists, rev_ppls, for_ppls)

        log_lstm_scores(bleus, selfbleus, dists, rev_ppls, for_ppls, args.log_file)

    # Case 8 : LSTM_VQ_VAE (Need more automization)
    elif args.model == 'lstmvqvae':        # mod : bc or other datasets
        train_loader = batchify(corpus.train, args.batch_size,shuffle=True, is_gpu=args.cuda)
        test_loader = batchify(corpus.test, args.batch_size,shuffle=False, is_gpu=args.cuda)

        Model = LSTM_VQ_VAE(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nemb = args.nemb,
                         nembdim = args.nembdim,
                         nlayers= args.nlayers,
                         word_dropout=args.word_dropout,
                         commit_cost=args.commit_cost,
                         init_method=args.init_method,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr_ae)

        bleus = np.array([])
        selfbleus = np.array([])
        dists = np.array([])
        rev_ppls = np.array([])
        for_ppls = np.array([])

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file, args.log_interval)
            ep_bleus = Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            ep_selfbleus, ep_dists = Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         log_file = args.log_file, save_path=args.save_path,
                                                   sample_method='greedy')
                                                   # sample_method='sampling')
            rev_ppl, for_ppl = train_lm(data_path=args.data_path, save_path=args.save_path, maxlen=args.maxlen,
                                        sample_method='greedy',
                                        dict=corpus.dictionary, N=5, epoch=epoch, model=Model, log_file=args.log_file)

            bleus, selfbleus, dists, rev_ppls, for_ppls = lstm_scores(ep_bleus, ep_selfbleus, ep_dists, rev_ppl,
                                                                      for_ppl,
                                                                      bleus, selfbleus, dists, rev_ppls, for_ppls)

        log_lstm_scores(bleus, selfbleus, dists, rev_ppls, for_ppls, args.log_file)

    # Case 9 : SNLI with LSTM-AE
    elif args.model == 'lstmae':  # mod : bc or other datasets
        Model = LSTM_AE(enc='lstm', dec='lstm',
                         nlatent=args.nlatent,
                         ntokens=ntokens,
                         nembdim=args.nembdim,
                         nlayers=args.nlayers,
                         nhidden=args.nhidden,
                         hidden_noise_r=args.noise_r,
                         is_gpu=args.cuda)

        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr_ae)

        bleus = np.array([])
        selfbleus = np.array([])
        dists = np.array([])
        rev_ppls = np.array([])
        for_ppls = np.array([])

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file, args.log_interval)
            ep_bleus = Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                                        args.save_path)
            ep_selfbleus, ep_dists = Model.sample(epoch, sample_num=args.sample_num, maxlen=args.maxlen,
                                                  idx2word=corpus.dictionary.idx2word,
                                                  log_file=args.log_file, save_path=args.save_path,
                                                  sample_method='greedy')
                                                  #sample_method='sampling')

            rev_ppl, for_ppl = train_lm(data_path=args.data_path, save_path=args.save_path, maxlen=args.maxlen,
                                        sample_method='greedy',
                                        dict=corpus.dictionary, N=5, epoch=epoch, model=Model, log_file=args.log_file)

            bleus, selfbleus, dists, rev_ppls, for_ppls = lstm_scores(ep_bleus, ep_selfbleus, ep_dists, rev_ppl,
                                                                      for_ppl,
                                                                      bleus, selfbleus, dists, rev_ppls, for_ppls)

        log_lstm_scores(bleus, selfbleus, dists, rev_ppls, for_ppls, args.log_file)

    # Case 10 : SNLI with LSTM_VQ_VAE2
    elif args.model == 'lstmvqvae2':        # mod : bc or other datasets
        Model = LSTM_VQ_VAE2(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nemb = args.nemb,
                         nembdim = args.nembdim,
                         nembdiml=args.nembdiml,
                         nlayers= args.nlayers,
                         word_dropout=args.word_dropout,
                         commit_cost=args.commit_cost,
                         init_method=args.init_method,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr_ae)

        bleus = np.array([])
        selfbleus = np.array([])
        dists = np.array([])
        rev_ppls = np.array([])
        for_ppls = np.array([])

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file, args.log_interval)
            ep_bleus = Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            ep_selfbleus, ep_dists = Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         log_file = args.log_file, save_path=args.save_path,
                                                   sample_method='greedy')
                                                   # sample_method='sampling')
            rev_ppl, for_ppl = train_lm(data_path=args.data_path, save_path=args.save_path, maxlen=args.maxlen,
                                        sample_method='greedy',
                                        dict=corpus.dictionary, N=5, epoch=epoch, model=Model, log_file=args.log_file)

            bleus, selfbleus, dists, rev_ppls, for_ppls = lstm_scores(ep_bleus, ep_selfbleus, ep_dists, rev_ppl,
                                                                      for_ppl,
                                                                      bleus, selfbleus, dists, rev_ppls, for_ppls)

        log_lstm_scores(bleus, selfbleus, dists, rev_ppls, for_ppls, args.log_file)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE code')

    # System Argument
    parser.add_argument('--gpu_num', type=int, default= 0, help='gpu number to run the model')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--seed', type=int, default=111152, help='random seed')
    parser.add_argument('--sample_num', type=int, default=20, help='The number of samples')

    # Data & Model Arguments
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset; [mnist, snli, bc]')
    parser.add_argument('--model', type=str, default='lstmvqvae', help='model; [vae, aae, arae, lstmvae, lstmaae,'
                                                                       'vqvae, lstmvqvae, lstmae]')
    parser.add_argument('--maxlen', type=int, default=30, help='Max length of the sentence; Exceeded words are truncated')

    # Model Architecture Arguments
    parser.add_argument('--ninput', type=int, default=784, help='The dimension size of input')
    parser.add_argument('--nembdim', type=int, default=300, help='The dimension size of embedding')
    parser.add_argument('--nemb', type=int, default=512, help='The number of embeddings for VQ VAE') # mod
    parser.add_argument('--nlatent', type=int, default=120, help='The dimension size of latent')
    parser.add_argument('--nlayers', type=int, default=1, help='The number of layers')
    parser.add_argument('--nhidden', type=int, default=1000, help='The hidden dimension size of LSTM or CNN')
    parser.add_argument('--nvocab', type=int, default=20000, help='The number of vocabulary to use')
    parser.add_argument('--nnoise', type=int, default=100, help='The dimension of noise for ARAE')
    parser.add_argument('--nDhidden', type=int, default=500, help='The hidden dimension size of Discriminator')
    parser.add_argument('--nGhidden', type=int, default=500, help='The hidden dimension size of Generator')
    parser.add_argument('--D_arch', type=str, default='300-300', help='The architechture of Discriminator')
    parser.add_argument('--G_arch', type=str, default='300-300', help='The architechture of Generator')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=20, help='The maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='The number of batch size')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='Learning rate of Autoencoder')
    parser.add_argument('--lr_adv', type=float, default=1e-4, help='Learning rate of encoder with adversarial loss')
    parser.add_argument('--lr_D', type=float, default=5e-5, help='Learning rate of Discriminator')
    parser.add_argument('--lr_G', type=float, default=5e-5, help='Learning rate of Generator')
    parser.add_argument('--anneal_function', type=str, default='logistic', help='kl annealing function; [logistic]') # Add other functions
    parser.add_argument('--commit_cost', type=float, default=0.25, help='Commitment cost for VQ VAE')
    parser.add_argument('--niters_ae', type=int, default=1, help='The number of iteration for ae')
    parser.add_argument('--niters_gan_d', type=int, default=1, help='The number of iteration for discriminator')
    parser.add_argument('--niters_gan_g', type=int, default=1, help='The number of iteration for generator')
    parser.add_argument('--niters_gan_ae', type=int, default=1, help='The number of iteration for ae using gan')
    parser.add_argument('--niters_gan_schedule', type=str, default='2-4-6', help='GAN SCHEDULE')
    parser.add_argument('--noise_r', type=float, default=0.05, help='stdev of noise for autoencoder (regularizer)')
    parser.add_argument('--noise_anneal', type=float, default=0.9995,
                        help='anneal noise_radius exponentially by this every 100 iterations')
    parser.add_argument('--gp', action='store_true', help='apply gradient penalty')
    parser.add_argument('--word_dropout', type=float, default=0.5, help='Word dropout of VAE model')

    # File load & Save  Arguments
    parser.add_argument('--save_path', type=str, default=None, help='location to save the trained file & samples')
    parser.add_argument('--log_interval', type=int, default=200, help='interval to log training results')

    # Utilty Arguments # mod : is it really need?
    parser.add_argument('--lowercase', action='store_true',help='lowercase all text')
    parser.add_argument('--config', type = int, default= None, help='default configuartion for each number')

    # VQ VAE argument
    parser.add_argument('--init_method', type=str, default='uniform', help='Initialize the latent')

    # Could add another arguments

    #{1 : VAE MNIST, 2: AAE MNIST, 3: ARAE MNIST, 4: VQ VAE MNIST, 5: LSTM VAE SNLI, 6: LSTM AAE SNLI, 7: LSTM ARAE SNLI, 8: LSTM VQ VAE SNLI}

    args = parser.parse_args()
    if args.config is not None:
        args = config_args(args, args.config)
    print(vars(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_num)

    # Make the save path
    if args.save_path is None:
        args.save_path = './outputs/{}/{}/'.format(args.model, args.dataset)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # Initialize the log file
    args.log_file = os.path.join(args.save_path, 'logs.txt')
    with open(args.log_file, 'w') as f:
        pass

    if args.dataset == 'mnist':
        args.data_path = './data'
    else:
        args.data_path = './data/' + args.dataset

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Cuda warnining
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, "
                  "so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    main(args)
