from src.data import Data

import torch

from torch import nn as nn
from torch.utils.data.dataloader import DataLoader as dl
from torch.utils.data.sampler import SubsetRandomSampler as srs

import argparse

# Optional args
parser = argparse.ArgumentParser()
parser.add_argument('-tp', type=int, default=0.8, metavar='PCT',
                    help='Train / Validation split (default: 0.8)')
parser.add_argument('-lr', type=float, default=0.001, metavar='LR',
                    help='Learning rate (default: 0.001)')
parser.add_argument('-e', type=int, default=1, metavar='N',
                    help="Number of epochs to train on training data.")
parser.add_argument('--cuda', type=bool, default=True, metavar='N')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='How many batches to wait before '
                         'logging training (default: 100)')
parser.add_argument('--optimizer', type=str, default='rms_prop',
                    choices=["rms_prop", "sgd", "adam"],
                    help="Which gradient descent algorithm to use.")
parser.add_argument('--loss_fn', type=str, default='bce', choices=["bce"],
                    help="Which loss function to use.")
parser.add_argument('--torch_seed', type=int, default=42,
                    help="Seed for torch randomization.")
args = parser.parse_args()

class Cetka(nn.Module):
    def __init__(self):
        super(Cetka).__init__()
        self.vocab_size
        self.num_layers
        self.num_hidden

        self.embedding = nn.Embedding
        self.lstm
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear()
        self.tanh = nn.Tanh()




if __name__ == '__main__':
    # Attempt to grab CUDA. Grab cpu if that fails
    use_cuda = args.cuda and torch.cuda.is_available()
    print(f"#\tCuda enabled: " + str(use_cuda))
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.torch_seed)

    # label conversion dictionary
    lookup = {'positive': 1, 'neutral': 0, 'negative': -1}

    # instantiate data set
    train = Data("../data/kaggle/", lookup)
    train_idx, val_idx = train.split_data(args.tp, shuffle=True)
    print(len(train_idx), len(val_idx))
    test = Data("../data/kaggle/", lookup, False)

    #for x, y, z in train:
    #    print(x)
