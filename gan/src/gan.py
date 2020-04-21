import argparse

import torchvision
from matplotlib.pyplot import imshow
from torchvision import datasets, transforms
from src.discriminator import Discriminator
from src.generator import Generator
import numpy as np
from matplotlib import pyplot as plt
import torch

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Neural Net to Model FF PUFs')
parser.add_argument('-dlr', type=float, default=0.0004, metavar='LR',
                    help='Learning rate for discriminator (default: 0.001)')
parser.add_argument('-glr', type=float, default=0.0004, metavar='LR',
                    help='Learning rate for generator (default: 0.001)')
parser.add_argument('--cuda', type=bool, default=True, help='Should the program use CUDA?')
args = parser.parse_args()


class GAN:
    def __init__(self, _device):
        self.device = _device
        self.batch_size = 64
        self.resolution = 28
        self.d_criterion = None
        self.d_optimizer = None
        self.g_criterion = None
        self.g_optimizer = None

        self.discriminator = Discriminator(num_layers=5,
                                           activations=["relu", "relu", "relu", "sigmoid"],
                                           device=_device,
                                           num_nodes=[1, 64, 64, 128, 1],
                                           kernels=[3, 3, 3],
                                           strides=[2, 2, 2],
                                           dropouts=[.25, .25, 0],
                                           batch_size=64)

        # pass one image through the network so as to initialize the ouput layer
        self.discriminator(torch.rand(size=[self.batch_size, 1, self.resolution, self.resolution]))

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

    def train(self, epochs: int, dataloader):
        for epoch in range(epochs):
            i = 0

            for data, target in dataloader:
                d_string = self.train_discriminator(data)
                self.train_generator()
                g_loss = self.train_generator()
                if i % 100 == 0:
                    print(f"@\tIndex: {i}\t" + d_string)
                    print(f"@\tGenerator Loss: {g_loss}")
                i += 1
        self.test()

    def train_discriminator(self, train_data):
        true = torch.ones((self.batch_size, 1)).to(self.device, dtype=torch.float64)
        false = torch.zeros((self.batch_size, 1)).to(self.device, dtype=torch.float64)

        index = np.random.randint(0, train_data.shape[0], self.batch_size)
        true_images = train_data[index]
        true_loss = self.discriminator.batch_train(true_images, true, self.d_criterion, self.d_optimizer)

        # FIXME: Extract 100 to argument
        noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, 1, 1, 100))).to(self.device, dtype=torch.float64)
        generated_images = self.generator(noise)
        false_loss = self.discriminator.batch_train(generated_images, false, self.d_criterion, self.d_optimizer)

        return f"True loss: {true_loss}\t False loss: {false_loss}"

    def train_generator(self):
        valid = torch.ones((self.batch_size, 1)).to(self.device, dtype=torch.float64)
        noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, 1, 1, 100))).to(self.device, dtype=torch.float64)
        return self.generator.batch_train(self.discriminator, noise, valid, self.g_criterion, self.g_optimizer)

    # make noise, and send through discriminator
    def test(self):
        noise = torch.tensor(np.random.normal(0, 1, (64, 1, 1, 100))).to(self.device, dtype=torch.float64)
        image = self.generator(noise).detach().cpu().numpy()
        for i in range(np.size(image, 0)):
            picture = image[i, 0,:,:]
            plt.imshow(picture)
            plt.show()




# note to self, Tensors are of form B C W H D
if __name__ == "__main__":
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    print(f"\tDevice selected: {device}")

    model = GAN(device)
    model.discriminator = model.discriminator.to(device)
    model.generator = model.generator.to(device)

    print(model.discriminator.parameters)
    print(model.generator.parameters)

    model.d_criterion = torch.nn.functional.binary_cross_entropy
    model.d_optimizer = torch.optim.RMSprop(params=model.discriminator.parameters(), lr=args.dlr)
    model.g_criterion = torch.nn.functional.binary_cross_entropy
    model.g_optimizer = torch.optim.RMSprop(params=model.generator.parameters(), lr=args.glr)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=model.batch_size, shuffle=True)

    model.train(1, train_loader)
