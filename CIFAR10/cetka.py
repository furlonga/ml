from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from CIFAR10.trainer import train, test

parser = argparse.ArgumentParser(description='Pytorch furlong Cifar10')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')


# Definition of the Neural Network Object.
# Input must be in form of B H W C
class Cetka(nn.Module):
    # define values and initialization protocols
    def __init__(self):
        super(Cetka, self).__init__()

        self.conv0 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(6, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 256, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv6(x))

        x = x.view(-1, 4 * 4 * 256)

        # print("shape = " + str(x.shape))

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:  # Get the products
            num_features *= s
        return num_features

    # statistics function, might be worth expanding
    def stat(self, x):
        m = x.mean()
        s = x.std()
        print(f"Mean: {m}, STD: {s}")


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = Cetka().to(device)
    model.share_memory()  # gradients are allocated lazily, so they are not shared here

    # TODO: Multithread
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device, dataloader_kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Once training is complete, we can test the model
    test(args, model, device, dataloader_kwargs)
