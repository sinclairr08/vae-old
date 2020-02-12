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

from utils import to_gpu, batchify
from preprocess import Corpus

def main(args):

    # Case 1 : MNIST with VAE
    if args.dataset == 'mnist' and args.model == 'vae':
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # mod : Is it really need?

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train = True, download=True,
                           transform = transforms.ToTensor()),
            batch_size=args.batch_size, shuffle = True, **kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, ** kwargs
        )

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

    # Case 2 : MNIST with AAE
    elif args.dataset == 'mnist' and args.model == 'aae':
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # mod : Is it really need?

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train = True, download=True,
                           transform = transforms.ToTensor()),
            batch_size=args.batch_size, shuffle = True, **kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, ** kwargs
        )

        Model = AAE(nlatent=args.nlatent,
                    ninput=args.ninput,
                    nhidden=args.nhidden,
                    nDhidden=args.nDhidden,
                    is_gpu = args.cuda)
        Model = to_gpu(Model, args.cuda)

        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc,
                              args.log_file)
            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)

    # Case 3 : MNIST with ARAE
    elif args.dataset == 'mnist' and args.model == 'arae':
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # mod : Is it really need?

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train = True, download=True,
                           transform = transforms.ToTensor()),
            batch_size=args.batch_size, shuffle = True, **kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, ** kwargs
        )

        Model = ARAE(nlatent=args.nlatent,
                    ninput=args.ninput,
                    nhidden=args.nhidden,
                    nDhidden=args.nDhidden,
                    nGhidden=args.nGhidden,
                    nnoise= args.nnoise,
                    is_gpu = args.cuda)
        Model = to_gpu(Model, args.cuda)

        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)
        optim_gen = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_G)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc, optim_gen,
                              args.log_file)
            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)

    # Case 4 : SNLI with LSTM-VAE
    elif args.dataset == 'snli' and args.model == 'lstmvae':        # mod : bc or other datasets
        corpus = Corpus('./data/snli',
                        maxlen=args.maxlen,
                        vocab_size=args.nvocab,
                        lowercase=args.lowercase)
        ntokens = len(corpus.dictionary.word2idx)

        train_loader = batchify(corpus.train, args.batch_size,shuffle=True, is_gpu=args.cuda)
        test_loader = batchify(corpus.test, args.batch_size,shuffle=False, is_gpu=args.cuda)

        Model = LSTM_VAE(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nembdim = args.nembdim,
                         nlayers= args.nlayers,
                         nhidden= args.nhidden,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr_ae)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file, args.log_interval)
            Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         save_path=args.save_path, sample_method = 'sampling')

    # Case 5 : SNLI with LSTM-AAE
    elif args.dataset == 'snli' and args.model == 'lstmaae':        # mod : bc or other datasets
        corpus = Corpus('./data/snli',
                        maxlen=args.maxlen,
                        vocab_size=args.nvocab,
                        lowercase=args.lowercase)
        ntokens = len(corpus.dictionary.word2idx)

        train_loader = batchify(corpus.train, args.batch_size,shuffle=True, is_gpu=args.cuda)
        test_loader = batchify(corpus.test, args.batch_size,shuffle=False, is_gpu=args.cuda)

        Model = LSTM_AAE(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nembdim = args.nembdim,
                         nlayers= args.nlayers,
                         nDhidden=args.nDhidden,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)

        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc,
                              args.log_file, args.log_interval)
            Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         save_path=args.save_path, sample_method = 'sampling')

    # Case 6 : SNLI with LSTM-ARAE
    elif args.dataset == 'snli' and args.model == 'lstmarae':        # mod : bc or other datasets
        corpus = Corpus('./data/snli',
                        maxlen=args.maxlen,
                        vocab_size=args.nvocab,
                        lowercase=args.lowercase)
        ntokens = len(corpus.dictionary.word2idx)

        train_loader = batchify(corpus.train, args.batch_size,shuffle=True, is_gpu=args.cuda)
        test_loader = batchify(corpus.test, args.batch_size,shuffle=False, is_gpu=args.cuda)

        Model = LSTM_ARAE(enc = 'lstm', dec= 'lstm',
                         nlatent= args.nlatent,
                         ntokens = ntokens,
                         nembdim = args.nembdim,
                         nlayers= args.nlayers,
                         nDhidden=args.nDhidden,
                         nGhidden=args.nGhidden,
                         nnoise= args.nnoise,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)


        optim_enc_nll = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_ae)
        optim_enc_adv = torch.optim.Adam(Model.encoder.parameters(), lr=args.lr_adv)
        optim_dec = torch.optim.Adam(Model.decoder.parameters(), lr=args.lr_ae)
        optim_disc = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_D)
        optim_gen = torch.optim.Adam(Model.disc.parameters(), lr=args.lr_G)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, train_loader, optim_enc_nll, optim_enc_adv, optim_dec, optim_disc, optim_gen,
                              args.log_file, args.log_interval)
            Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         save_path=args.save_path, sample_method = 'sampling')

    # Case 7 : MNIST with VQ VAE (Need more automization)
    if args.dataset == 'mnist' and args.model == 'vqvae':
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # mod : Is it really need?

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train = True, download=True,
                           transform = transforms.ToTensor()),
            batch_size=args.batch_size, shuffle = True, **kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, ** kwargs
        )

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
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset; [mnist, snli]')
    parser.add_argument('--model', type=str, default='vqvae', help='model; [vae, aae, arae, lstmvae, lstmaae, vqvae]')
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

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=20, help='The maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='The number of batch size')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='Learning rate of Autoencoder')
    parser.add_argument('--lr_adv', type=float, default=5e-5, help='Learning rate of encoder with adversarial loss')
    parser.add_argument('--lr_D', type=float, default=5e-5, help='Learning rate of Discriminator')
    parser.add_argument('--lr_G', type=float, default=5e-5, help='Learning rate of Generator')
    parser.add_argument('--anneal_function', type=str, default='logistic', help='kl annealing function; [logistic]') # Add other functions
    parser.add_argument('--commit_cost', type=float, default=0.25, help='Commitment cost for VQ VAE')



    # File load & Save  Arguments
    parser.add_argument('--save_path', type=str, default=None, help='location to save the trained file & samples')
    parser.add_argument('--log_interval', type=int, default=200, help='interval to log training results')

    # Utilty Arguments # mod : is it really need?
    parser.add_argument('--lowercase', action='store_true',help='lowercase all text')

    # Could add another arguments


    # GUIDELINE for best hparams
    # VAE - MNIST HYPARAM
    # python3 main.py --model vae --dataset mnist --ninput 784 --nlatent 20 --nhidden 400 --lr_ae 1e-3

    # AAE - MNIST HYPARAM
    # python3 main.py --model aae --dataset mnist --ninput 784 --nlatent 120 --nhidden 1000 --nDhidden 500
    # lr ae = 1e-04 lr disc 5e-05

    # ARAE - MNIST HYPARAM
    # python3 main.py --model arae --dataset mnist --ninput 784 --nlatent 120 --nhidden 1000  --nDhidden 500 --nGhidden 500

    # LSTM VAE - SNLI HYPARAM
    # python3 main.py --model lstmvae --dataset snli --nlatent 300 --nhidden 300

    # LSTM AAE - SNLI HYPARAM
    # python3 main.py --model lstmaae --dataset snli --nlatent 300  --nDhidden 300

    # LSTM ARAE - SNLI HYPARAM
    # python3 main.py --model lstmarae --dataset snli --nlatent 300 --nnoise 100  --nDhidden 300 --nGhidden 300

    # VQ VAE - MNIST HYPARAM
    # python3 python3 main.py --model vqvae --dataset mnist --nlatent 64 --nembdim 64 --nemb 512 --ninput 784 --nhidden 400 --lr_ae 1e-3 --commit_cost 1

    args = parser.parse_args()
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
