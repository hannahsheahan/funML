# Build a simple autoencoder
# Author: Hannah Sheahan, sheahan.hannah@gmail.com
# Date: 13/05/2020
# Issues: N/A
# Notes: N/A
# ---------------------------------------------------------------------------- #

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse

# ---------------------------------------------------------------------------- #

class argsparser():
    # We wont bother with command line network parameters so just define them here
    def __init__(self):
        self.batch_size_train = 24
        self.batch_size_test = 24
        self.epochs = 50
        self.lr = 0.0001
        self.momentum = 0.9
        self.log_interval = 1000
        self.hidden_size = 40
        self.save_model = False
        self.no_cuda = False

# ---------------------------------------------------------------------------- #

def defineHyperparams():
    """This will enable us to take different network training settings/hyperparameters in when we call main.py from the command line.
    e.g. python3 main.py --batch-size=12 --epochs=20 --save-model
    Or if you want it to execute train/test across multiple combinations of hyperparameters, you can define multiple as follows:
    e.g. python3 main.py --batch-size-multi 12 24 48 --lr-multi 0.01 0.001"""
    args = argsparser()
    use_cuda = not args.no_cuda and torch.cuda.is_available() # use cuda/gpu if you can
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    parser = argparse.ArgumentParser(description='PyTorch network settings')

    # network training hyperparameters
    parser.add_argument('--batch-size-train', type=int, default=24, metavar='N', help='input batch size for training (default: 24)')
    parser.add_argument('--batch-size-test', type=int, default=24, metavar='N', help='input batch size for testing (default: 24)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', help='For Saving the current Model')
    parser.add_argument('--hidden-size', type=int, default=200, metavar='N', help='number of nodes in hidden layer (default: 60)')
    parser.add_argument('--model-id', type=int, default=0, metavar='N', help='for distinguishing many iterations of training same model (default: 0).')
    args = parser.parse_args()

    return args, device, kwargs

# ---------------------------------------------------------------------------- #

def train(model, device, criterion, optimizer, train_loader):

    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (input, _) in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
        flat_input = input.view(input.shape[0], 1, input.shape[2]*input.shape[3])
        flat_input, target = flat_input.to(device), flat_input.to(device)  # because target == input
        output = model(flat_input)            # forward pass
        loss = criterion(output, target)  # compute loss
        loss.backward()                  # backwards pass
        optimizer.step()                 # update weights

        train_loss += loss

    return train_loss

# ---------------------------------------------------------------------------- #

def test(model, device, criterion, test_loader):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(test_loader):
            flat_input = input.view(input.shape[0], 1, input.shape[2]*input.shape[3])
            flat_input, target = flat_input.to(device), flat_input.to(device)  # because target == input
            output = model(flat_input)            # forward pass
            loss = criterion(output, flat_input) # compute loss

            test_loss += loss

    return test_loss

# ---------------------------------------------------------------------------- #

class autoencoder(nn.Module):
    # a simple autoencoder that has only one hidden layer (bottleneck)
    def __init__(self, D_in, D_hidden):
        super(autoencoder, self).__init__()  # inherit the default initialisations from nn.Module too
        self.input_size = D_in
        self.hidden_size = D_hidden
        self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        self.latent_states = F.relu(self.encoder(x))
        self.output = F.relu(self.decoder(self.latent_states))
        return self.output

    def get_activations(self, x):
        # at test we get see how the latent states look for a given input
        self.forward(x)
        return self.latent_states, self.output

# ---------------------------------------------------------------------------- #

def main():

    # Define network training parameters
    args, device, kwargs = defineHyperparams()

    # Load MNIST data
    norm_mean, norm_std = [0.1307, 0.3081] # for normalizing the MNIST inputs (conventional but dunno what it means)
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(),
                                transforms.Normalize((norm_mean,), (norm_std,)) ]))
    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([ transforms.ToTensor(),
                                transforms.Normalize((norm_mean,), (norm_std,)) ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=True, **kwargs)

    # Initialise our model
    flat_MNIST_size = 28*28
    model = autoencoder(flat_MNIST_size, args.hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print("Training network...")
    for epoch in range(args.epochs):
        train_loss = train(model, device, criterion, optimizer, train_loader)
        test_loss = test(model, device, criterion, test_loader)
        print('Epoch {}:  Train loss: {}%, Test loss: {}%'.format(epoch, train_loss, test_loss))

    print("Training complete.")
    return model

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()
