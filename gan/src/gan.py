import argparse

from torchvision import datasets, transforms
from src.discriminator import Discriminator
from src.generator import Generator
import numpy as np
from matplotlib import pyplot as plt
import torch
import pdb

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Neural Net to Model FF PUFs')
parser.add_argument('-dlr', type=float, default=0.0008, metavar='LR',
                    help='Learning rate for discriminator (default: 0.001)')
parser.add_argument('-glr', type=float, default=0.0001, metavar='LR',
                    help='Learning rate for generator (default: 0.001)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Should the program use CUDA?')
args = parser.parse_args()

# arrays to store run data
t_loss = []
f_loss = []
g_loss = []


def wasserstein_loss(prediction, target):
    raise NotImplementedError


"""

"""

class GAN:
    def __init__(self, _device):

        self.device = _device
        self.batch_size = 256
        self.resolution = 28

        self.d_criterion = None
        self.g_criterion = None
        self.d_optimizer = None
        self.g_optimizer = None

        # declare the discriminator
        self.discriminator = Discriminator(num_layers=4,
                                           activations=["relu", "relu", "relu",
                                                        "sigmoid"],
                                           device=_device,
                                           num_nodes=[1, 64, 128, 1],
                                           kernels=[5, 5, 3],
                                           strides=[2, 2, 2],
                                           dropouts=[.25, .25, 0],
                                           batch_size=self.batch_size)
        # pass one image through the network so as to initialize the output
        # layer
        self.discriminator(torch.rand(
            size=[self.batch_size, 1, self.resolution, self.resolution]))

        # declare the generator
        self.generator = Generator(num_layers=5,
                                   activations=["relu", "relu", "relu", "relu",
                                                "tanh"],
                                   num_nodes=[1, 64, 128, 256, 1],
                                   kernels=[3, 3, 3, 3],
                                   strides=[1, 1, 1, 1],
                                   batch_norms=[1, 1, 1, 0],
                                   upsamples=[1, 1, 0],
                                   dropouts=[.25, .25, 0],
                                   batch_size=self.batch_size)

    def train(self, epochs: int, dataloader):
        for epoch in range(epochs):

            initial_loss = self.train_generator()
            self.display_output()
            false_loss = 1.
            print(f"#\tInitial Generator Loss: {initial_loss}")
            for i, (data, _) in enumerate(dataloader):
                # train limiter allowing powerful discriminator without
                # overpowering the generator
                if false_loss > 0.6:
                    true_loss, false_loss = self.train_discriminator(data, True)
                else:
                    true_loss, false_loss = self.train_discriminator(data,
                                                                     False)
                generator_loss = self.train_generator()

                if false_loss < 0:
                    return
                if i % 1 == 0:
                    print(
                        f"@\tIndex: {i}\tTrue Loss: {true_loss}\t"
                        f"False Loss: {false_loss}\t"
                        f"Generator Loss: {generator_loss}")
                    # log data for later graphing
                    #t_loss.append(true_loss)
                    #f_loss.append(false_loss)
                    #g_loss.append(generator_loss)

                    self.display_output(epoch, i)
        self.test()

    def train_discriminator(self, train_data, train):

        # add noise to labels
        true = torch.ones((self.batch_size, 1))
        noise = torch.nn.functional.relu(0.1 * torch.randn(self.batch_size, 1))
        true.sub_(noise)
        true = true.to(self.device, dtype=torch.float64)

        false = torch.zeros((self.batch_size, 1))
        noise = torch.nn.functional.relu(0.1 * torch.randn(self.batch_size, 1))
        false.sub_(noise)
        false = false.to(self.device, dtype=torch.float64)

        index = np.random.randint(0, train_data.shape[0], self.batch_size)
        true_images = train_data[index]
        true_loss = self.discriminator.batch_train(true_images, true,
                                                   self.d_criterion,
                                                   self.d_optimizer, train)

        # FIXME: Extract 100 to argument
        noise = torch.randn(self.batch_size, 1, 1, 100, dtype=torch.float64).to(
            self.device)
        generated_images = self.generator(noise)
        false_loss = self.discriminator.batch_train(generated_images, false,
                                                    self.d_criterion,
                                                    self.d_optimizer, train)

        return true_loss, false_loss

    def train_generator(self):
        valid = torch.ones((self.batch_size, 1), dtype=torch.float64).to(
            self.device)
        noise = torch.randn(self.batch_size, 1, 1, 100, dtype=torch.float64).to(
            self.device)
        return self.generator.batch_train(self.discriminator, noise, valid,
                                          self.g_criterion, self.g_optimizer)

    # make noise, and send through discriminator
    def test(self):
        noise = torch.randn(self.batch_size, 1, 1, 100, dtype=torch.float64).to(
            self.device)
        image = self.generator(noise).detach().cpu().numpy()
        for i in range(np.size(image, 0)):
            picture = image[i, 0, :, :]
            plt.imshow(picture)
            plt.show()

    def display_output(self, epoch=0, index=0):
        noise = torch.randn(1, 1, 1, 100, dtype=torch.float64).to(self.device)
        image = self.generator(noise).detach().cpu().numpy()
        picture = image[0, 0, :, :]
        plt.imshow(picture)
        plt.title(f"Epoch {epoch}, Index {index}")
        plt.show()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight)
        m.bias.data.fill_(0.01)


# note to self, Tensors are of form B C W H D
if __name__ == "__main__":
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    print(f"\tDevice selected: {device}")

    model = GAN(device)
    model.discriminator = model.discriminator.to(device)
    model.generator = model.generator.to(device)
    model.discriminator.apply(init_weights)
    model.generator.apply(init_weights)

    print(model.discriminator.parameters)
    print(model.generator.parameters)

    model.d_criterion = torch.nn.functional.binary_cross_entropy
    model.d_optimizer = torch.optim.RMSprop(
        params=model.discriminator.parameters(), lr=args.dlr)

    model.g_criterion = torch.nn.functional.binary_cross_entropy
    model.g_optimizer = torch.optim.Adam(params=model.generator.parameters(),
                                         lr=args.glr)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=model.batch_size, shuffle=True)

    model.train(100, train_loader)
