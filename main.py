import argparse
import os

import random
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms

from models import VAE, LSTM_VAE, AAE

from utils import to_gpu, batchify
from preprocess import Corpus

def main(args):

    # Case 1 : MNIST with VAE (Need more automization)
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
        optimizer = optim.Adam(Model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file)
            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)  ## MODIFICATION

    # Case 2 : SNLI with LSTM-VAE (Need more automization)
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
                         nemb = args.nemb,
                         nlayers= args.nlayers,
                         nhidden= args.nhidden,
                         is_gpu = args.cuda)

        Model = to_gpu(Model, args.cuda)
        optimizer = optim.Adam(Model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, optimizer, train_loader, args.log_file, args.log_interval)
            Model.test_epoch(epoch, test_loader, corpus.dictionary.idx2word, args.log_file,
                             args.save_path)
            Model.sample(epoch, sample_num=args.sample_num, maxlen = args.maxlen, idx2word = corpus.dictionary.idx2word,
                         save_path=args.save_path, sample_method = 'sampling')
            # mod : NAN error
            # moded : error fix

    # Case 3 : MNIST with AAE (Need more automization)
    if args.dataset == 'mnist' and args.model == 'aae':
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
                    is_gpu = args.cuda)
        Model = to_gpu(Model, args.cuda)

        for epoch in range(1, args.epochs + 1):
            Model.train_epoch(epoch, train_loader, args.log_file)
            Model.test_epoch(epoch, test_loader, args.log_file)
            Model.sample(epoch, sample_num=args.sample_num, save_path=args.save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE code')

    # System Argument
    parser.add_argument('--gpu_num', type=int, default= 0, help='gpu number to run the model')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--seed', type=int, default=111152, help='random seed')
    parser.add_argument('--sample_num', type=int, default=20, help='The number of samples')

    # Data & Model Arguments
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset; [mnist, snli]')
    parser.add_argument('--model', type=str, default='aae', help='model; [vae, lstmvae, aae]')
    parser.add_argument('--maxlen', type=int, default=30, help='Max length of the sentence; Exceeded words are truncated')

    # Model Architecture Arguments
    parser.add_argument('--ninput', type=int, default=784, help='The dimension size of input')
    parser.add_argument('--nemb', type=int, default=300, help='The dimension size of embedding')
    parser.add_argument('--nlatent', type=int, default=300, help='The dimension size of latent')
    parser.add_argument('--nlayers', type=int, default=1, help='The number of layers')
    parser.add_argument('--nhidden', type=int, default=300, help='The hidden dimension size of LSTM or CNN')
    parser.add_argument('--nvocab', type=int, default=20000, help='The number of vocabulary to use')
    # Can add - encoder arch, dec arch,

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=10, help='The maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='The number of batch size')
    parser.add_argument('--lr', type=float, default=1, help='Learning rate of ##') # Need separation.. and modification
    parser.add_argument('--anneal_function', type=str, default='logistic', help='kl annealing function; [logistic]') # Add other functions

    # File load & Save  Arguments
    parser.add_argument('--save_path', type=str, default=None, help='location to save the trained file & samples')
    parser.add_argument('--log_interval', type=int, default=200, help='interval to log training results')

    # Utilty Arguments # mod : is it really need?
    parser.add_argument('--lowercase', action='store_true',help='lowercase all text')

    # Could add another arguments

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
