import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class Generator(nn.Module):
    def __init__(self, num_layers, num_nodes, activations, kernels,
                 strides, upsamples, batch_norms, dropouts, batch_size):
        super(Generator, self).__init__()

        self.layers = nn.ModuleList([])
        # num layers is input/output inclusive
        self.num_layers = num_layers
        # num_nodes represents the nodes at each layer of the network
        # and is of size num_layers
        self.num_nodes = num_nodes
        self.num_conv = 0
        # of size num_layers - 1
        self.activations = activations
        self.kernels = kernels
        self.strides = strides
        self.upsamples = upsamples

        # function declarations
        self.upsample = nn.Upsample(scale_factor=2)
        # construct batch_norms
        self.batch_norms = []
        for index, value in enumerate(batch_norms):
            if value != 0:
                self.batch_norms.append(nn.BatchNorm2d(
                    num_features=self.num_nodes[index + 2]).cuda())
            else:
                self.batch_norms.append(0)
        # of size num_layers - 2
        self.dropouts = dropouts

        # append prefix layers
        self.linear_up = nn.Linear(100, 7 * 7 * 64)
        self.lu_relu = nn.ReLU()
        self.lu_bn = nn.BatchNorm2d(num_features=1)

        # append convolutional layers
        for i in range(1, self.num_layers - 1):
            self.layers.append(nn.Conv2d(self.num_nodes[i],
                                         self.num_nodes[i + 1],
                                         padding=True,
                                         kernel_size=self.kernels[i - 1],
                                         stride=self.strides[i - 1]))
            self.num_conv += 1

        # Asserts to ensure legal parameters were entered
        assert not len(self.kernels) == len(self.strides) != len(
            self.activations) - 1, "mismatch on module parameters"
        assert not len(self.activations) == self.num_layers - 1 != len(
            self.num_nodes) - 1, "mismatch on module parameters"

    def forward(self, x):
        # find the batch size during a forward pass
        batch_size = x.shape[0]

        # Take the noise and convert to a 2d tensor.
        x = self.linear_up(x)
        x = self.lu_bn(x)
        x = self.lu_relu(x)

        # reshape
        x = x.view(batch_size, self.num_nodes[1], 7, 7)

        for index in range(self.num_conv):

            # upsample, if allowed
            if self.upsamples[index] == 1:
                x = self.upsample(x)
            # convolve
            x = self.layers[index](x)

            # batch norm, if allowed
            if self.batch_norms[index] == 1:
                x = self.batch_norms[index](x)

            # activation function
            x = self.activation(self.activations[index])(x)
        return x

    def batch_train(self, discriminator, train_batch, targets,
                    criterion, optimizer):
        # This is one epoch of training the discriminator.
        # This is called from GAN. Targets are manually supplied.
        self.train()
        optimizer.zero_grad()
        # Pass the batch through the model (CUDA)
        generated = self(train_batch)
        prediction = discriminator(generated)
        # Calculate loss
        loss = criterion(prediction, targets)

        loss.backward()
        optimizer.step()
        self.eval()
        return loss

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:  # Get the products
            num_features *= s
        return num_features

    @staticmethod
    def activation(name):
        if name == "relu":
            return torch.nn.LeakyReLU()
        if name == "sigmoid":
            return torch.sigmoid
        if name == "tanh":
            return torch.tanh
