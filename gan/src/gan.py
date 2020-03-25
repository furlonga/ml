from src.discriminator import Discriminator
from src.generator import Generator
import numpy as np
import torch


class GAN:
    def __init__(self):
        self.discriminator = Discriminator(num_layers=5,
                                           activations=["relu", "relu", "relu", "sigmoid"],
                                           num_nodes=[1, 64, 64, 128, 1],
                                           kernels=[3, 3, 3],
                                           strides=[2, 2, 2],
                                           dropouts=[.25, .25, 0],
                                           batch_size=1)

        self.generator = Generator(num_layers=5,
                                   activations=["relu", "relu", "relu", "sigmoid"],
                                   num_nodes=[1, 128, 64, 64, 1],
                                   kernels=[3, 3, 3],
                                   strides=[2, 2, 2],
                                   dropouts=[.25, .25, 0],
                                   batch_size=1)


if __name__ == "__main__":
    gan = GAN()
    print(gan.discriminator)
    print(gan.discriminator(torch.rand(size=[1, 1, 28, 28])))
