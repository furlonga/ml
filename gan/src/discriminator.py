import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


# TODO: Add device agnostic for cuda
#       Additionally,
class Discriminator(nn.Module):
    def __init__(self, num_layers, num_nodes, device, activations, kernels,
                 strides, dropouts, batch_size):

        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([])
        self.device = device
        self.batch_size = batch_size
        # num layers is input/output inclusive
        self.num_layers = num_layers
        self.output_layer = None
        # num_nodes represents the nodes at each layer of the network and is
        # of size num_layers
        self.num_nodes = num_nodes
        self.num_conv = 0
        # of size num_layers - 1
        self.activations = activations
        self.kernels = kernels
        self.strides = strides

        # of size num_layers - 2
        self.dropouts = dropouts

        # append layers
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Conv2d(self.num_nodes[i],
                                         self.num_nodes[i + 1],
                                         padding=True,
                                         kernel_size=self.kernels[i],
                                         stride=self.strides[i]))
            self.num_conv += 1

        # Asserts to ensure legal parameters were entered
        assert len(self.kernels) == len(self.strides) == len(
            self.activations) - 1, "mismatch on module parameters"
        assert not len(self.activations) == self.num_layers - 1 != len(
            self.num_nodes) - 1, "mismatch on module parameters"

    def forward(self, x):
        for index in range(self.num_conv):
            # take information from block on convolution
            x = self.layers[index](x)
            x = self.activation(self.activations[index])(x)
            # dropout of size activation, with 0 denoting shortcut
            x = torch.dropout(x, self.dropouts[index], self.training)

        # reshape the tensor
        x = x.view(self.batch_size, -1)

        # Check to see if there is a output layer predefined.
        if len(self.layers) == self.num_conv:
            self.layers.append(nn.Linear(self.num_flat_features(x), 1))

        x = self.layers[-1](x)
        x = self.activation(self.activations[-1])(x)
        return x

    def batch_train(self, train_batch, targets, criterion, optimizer, train):
        # This is one epoch of training the discriminator.
        # This is called from GAN. Targets are manually supplied.
        if train:
            self.train()
        else:
            self.eval()

        optimizer.zero_grad()
        # Pass the batch through the model (CUDA)
        prediction = self(train_batch.to(self.device, dtype=torch.float64))
        # Calculate loss
        loss = criterion(prediction, targets)

        if train:
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
            return torch.nn.functional.leaky_relu
        if name == "sigmoid":
            return torch.sigmoid
        if name == "tanh":
            return torch.tanh

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
