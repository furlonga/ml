import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_layers, num_nodes, activations, kernels, strides, dropouts, batch_size):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList([])
        # num layers is input/output inclusive
        self.num_layers = num_layers
        # num_nodes represents the nodes at each layer of the network and is of size num_layers
        self.num_nodes = num_nodes
        self.num_conv_t = 0
        # of size num_layers - 1
        self.activations = activations
        self.kernels = kernels
        self.strides = strides
        # of size num_layers - 2
        self.dropouts = dropouts
        self.training = False

        # append layers
        for i in range(self.num_layers - 2):
            self.layers.append(nn.ConvTranspose2d(self.num_nodes[i],
                                                  self.num_nodes[i + 1],
                                                  padding=True,
                                                  kernel_size=self.kernels[i],
                                                  stride=self.strides[i]))
            self.num_conv_t += 1

        self.layers.append(nn.Linear(self.num_nodes[-2], 1))

        assert len(self.kernels) == len(self.strides) \
               == len(self.activations) - 1, "mismatch on module parameters"
        assert len(self.activations) == self.num_layers - 1 \
               == len(self.num_nodes) - 1, "mismatch on module parameters"

    def forward(self, x):
        for index in range(self.num_conv_t):
            # take information from block on convolution
            x = self.layers[index](x)
            x = self.activation(self.activations[index])(x)
            # dropout of size activation, with 0 denoting shortcut
            x = torch.dropout(x, self.dropouts[index], self.training)

        # reshape the tensor
        x = x.view(1, -1)

        # use of the word convolutions is confusing here
        x = self.layers[-1](x)
        x = self.activation(self.activations[-1])(x)
        return x

    @staticmethod
    def activation(name):
        if name == "relu":
            return torch.relu
        if name == "sigmoid":
            return torch.sigmoid
        if name == "tanh":
            return torch.tanh
