import argparse
import os

import random
import numpy as np
import torch

def main(sth):
    print("WOW")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE code')

    # System Argument
    parser.add_argument('--gpu_num', type=int, default= 0, help='gpu number to run the model')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--seed', type=int, default=111152, help='random seed')

    # Data & Model Arguments
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset; [mnist]')

    # Model Architecture Arguments
    parser.add_argument('--nlatent', type=int, default=300, help='The dimension size of latent')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=50, help='The maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of batch size')
    parser.add_argument('--lr', type=float, default=1e-01, help='Learning rate of ##') # Need separation
    parser.add_argument('--anneal_function', type=str, default='logistic', help='kl annealing function; [logistic]') # Add other functions

    # File load & Save
    parser.add_argument('--save_path', type=str, default=None, help='location to save the trained file & samples')

    # Could add another arguments
    #

    args = parser.parse_args()
    print(vars(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_num)

    # We have to add more auto-completion of save folder name
    save_path = args.save_path
    if save_path is None:
        save_path = './outputs/VAE'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, "
                  "so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    main(args)
