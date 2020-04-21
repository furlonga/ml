import argparse

from src.discriminator import Discriminator
from src.generator import Generator

import numpy as np
import torch

parser = argparse.ArgumentParser(description='Neural Net to Model FF PUFs')
parser.add_argument('-dlr', type=float, default=0.001, metavar='LR',
                    help='Learning rate for discriminator (default: 0.001)')
parser.add_argument('-glr', type=float, default=0.001, metavar='LR',
                    help='Learning rate for generator (default: 0.001)')
parser.add_argument('--cuda', type=bool, default=True, help='Should the program use CUDA?')
args = parser.parse_args()


class GAN:
    def __init__(self, _device):
        self.discriminator = Discriminator(num_layers=5,
                                           activations=["relu", "relu", "relu", "sigmoid"],
                                           device=_device,
                                           num_nodes=[1, 64, 64, 128, 1],
                                           kernels=[3, 3, 3],
                                           strides=[2, 2, 2],
                                           dropouts=[.25, .25, 0])

        self.generator = Generator(num_layers=6,
                                   activations=["relu", "relu", "relu", "relu", "tanh"],
                                   device=_device,
                                   num_nodes=[1, 64, 128, 64, 64, 1],
                                   kernels=[3, 3, 3, 3],
                                   strides=[1, 1, 1, 1],
                                   batch_norms=[1, 1, 1, 0],
                                   upsamples=[True, True, False, False],
                                   dropouts=[.25, .25, 0],
                                   batch_size=64)

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.train_epoch()
            self.test()

    def train_epoch(self):
        self.train_discriminator()
        self.train_generator()

    def train_discriminator(self, train_data, batch_size):
        true = np.ones((batch_size, 1))
        false = np.zeros((batch_size, 1))

        index = np.random.randint(0, train_data.shape[0], batch_size)
        true_images = train_data[index]
        self.discriminator.batch_train(true_images, true)

    def train_generator(self):
        return None

    # make noise, and send through discriminator
    def test(self):
        return None


if __name__ == "__main__":
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    print(f"\tDevice selected: {device}")

    model = GAN(device)

    d_criterion = torch.nn.functional.binary_cross_entropy
    #    d_optimizer = torch.optim.RMSprop(params=model.discriminator.parameters, lr=args.dlr)
    g_criterion = torch.nn.functional.binary_cross_entropy
    #   g_optimizer = torch.optim.RMSprop(params=model.generator.parameters, lr=args.glr)

    print(model.discriminator)
    print(model.discriminator(torch.rand(size=[1, 1, 28, 28])))
    print(model.generator)
    print(model.generator(torch.rand(size=(64, 1, 1, 100))).shape)
