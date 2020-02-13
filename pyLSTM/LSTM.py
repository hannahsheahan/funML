"""
This is a toy project for learning the basics of LSTMs and implementing one in python.

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 13/02/2020
Issues: N/A
Notes: N/A

"""
# ---------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import sys

# pytorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse

# ---------------------------------------------------------------------------- #

def printProgress(i, numiter):
    """This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""
    j = (i+1)/numiter
    sys.stdout.write('r/')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()

# ---------------------------------------------------------------------------- #

def defineHyperparams():
    """
    This will enable us to take different network training settings/hyperparameters in when we call main.py from the command line.
    e.g. python3 main.py --batch-size=12 --epochs=20 --save-model
    Or if you want it to execute train/test across multiple combinations of hyperparameters, you can define multiple as follows:
    e.g. python3 main.py --batch-size-multi 12 24 48 --lr-multi 0.01 0.001
    If you are running this from a notebook and not the command line, just adjust the params specified in the class argparser()
    """
    args = argsparser()
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # do parallel computing if possible
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    command_line = True  # if running from jupyter notebook, set this to false and adjust argsparser() instead
    if command_line:
        parser = argparse.ArgumentParser(description='PyTorch network settings')
        parser.add_argument('--modeltype', default="aggregate", help='input type for selecting which network to train (default: "aggregate", concatenates pixel and location information)')
        parser.add_argument('--batch-size-multi', nargs='*', type=int, help='input batch size (or list of batch sizes) for training (default: 48)', default=[12])
        parser.add_argument('--lr-multi', nargs='*', type=float, help='learning rate (or list of learning rates) (default: 0.001)', default=[0.001])
        parser.add_argument('--batch-size', type=int, default=12, metavar='N', help='input batch size for training (default: 48)')
        parser.add_argument('--test-batch-size', type=int, default=12, metavar='N', help='input batch size for testing (default: 48)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--weight_decay', type=int, default=0.0000, metavar='N', help='weight-decay for l2 regularisation (default: 0)')
        parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
        args = parser.parse_args()

    multiparams = [args.batch_size_multi, args.lr_multi]
    return args, device, multiparams

# ---------------------------------------------------------------------------- #

class argsparser():
    """For holding network training arguments."""
    def __init__(self):
        self.batch_size = 24
        self.test_batch_size = 24
        self.epochs = 50
        self.lr = 0.002
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 1000
        self.weight_decay = 0.00
        self.save_model = False

# ---------------------------------------------------------------------------- #

def trainLSTM():
    # my LSTM training function goes here...
    pass


# ---------------------------------------------------------------------------- #

def testLSTM():
    # my LSTM testing function goes here...
    pass

# ---------------------------------------------------------------------------- #

def defineDataset():
    # load in a dataset, split into train/test and set up some dataloaders so that we can batch train/test
    pass

# ---------------------------------------------------------------------------- #

class myLSTM(nn.Module):
    """ To be implemented later """
    def __init__(self):
        pass

    self.somevariable = 0


# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    # execute our main process here so that it runs train/test etc from command line,
    # but doesn't automatically if we import it

    pass

# ---------------------------------------------------------------------------- #
