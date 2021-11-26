import random

from src.language import Language
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import argparse

# Optional args
parser = argparse.ArgumentParser()
parser.add_argument('-tp', type=int, default=0.8, metavar='PCT',
                    help='Train / Validation split (default: 0.8)')
parser.add_argument('-lr', type=float, default=0.0008, metavar='LR',
                    help='Learning rate (default: 0.008)')
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

teacher_forcing_ratio = 0.5


# TODO:
#   Batch
#   GloVe
#   RoBERTa / BERT


# definition of the encoder rnn
class EncoderRNN(nn.Module):
    def __init__(self, input_size, _hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_Size = input_size
        self.hidden_size = _hidden_size

        self.embedding = nn.Embedding(input_size, _hidden_size)
        self.gru = nn.GRU(_hidden_size, _hidden_size)

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# definition of the decoder rnn
class DecoderRNN(nn.Module):
    def __init__(self, _hidden_size, output_size, dropout=0.2, length=100):
        super(DecoderRNN, self).__init__()
        self.hidden_size = _hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.length = length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.length)
        self.attention_c = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = torch.softmax(
            self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attention_c(output).unsqueeze(0)

        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Modified heavily for this specific use case.
def train_iteration(input_tensor, target_tensor, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion, data,
                    max_length=100):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = input_tensor[0]
    target_tensor = target_tensor[0]

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    prediction = ''
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                  device=device)

    loss = 0
    # encode input and prepare for decoder
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[1]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() \
                                  < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            target = torch.tensor([target_tensor[di]], device=device,
                                  dtype=torch.int64)
            loss += criterion(decoder_output, target)

            decoder_input = target_tensor[di]  # Teacher forcing
            prediction += data.int_to_token[decoder_output.max(1)[1].item()] \
                          + " "
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            target = torch.tensor([target_tensor[di]], device=device,
                                  dtype=torch.int64)

            loss += criterion(decoder_output, target)
            prediction += data.int_to_token[decoder_output.max(1)[1].item()] \
                          + " "
            if decoder_input.item() == "</s>":
                break

    # gradient descent
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # return loss, target, prediction, input
    # TODO: No need to convert from int back to token.
    return loss.item() / target_length, tensor_to_words(target_tensor, data), \
           prediction, tensor_to_words(input_tensor, data)


def train(data, _train_loader, _val_loader, encoder, decoder, epochs):
    e_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    d_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    # criterion = jaccard_distance_loss
    criterion = nn.CrossEntropyLoss()

    for e in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        for i, (x, y) in enumerate(_train_loader):
            if x.size(1) < 2 or y.size(1) < 1:
                continue
            x = x.to(device)
            y = y.to(device)
            loss, string, guess, original = train_iteration(x, y, encoder,
                                                            decoder,
                                                            e_optimizer,
                                                            d_optimizer,
                                                            criterion,
                                                            data)
            if i % 10 == 0:
                prediction = process_prediction(original, guess)
                print(
                    f"@\tIndex: {i}\tLoss: {loss:.3f}\t\tJaccard: "
                    f"{jaccard(string, guess):.3f}\tAugmented Jaccard: "
                    f"{jaccard(string, prediction):.3f} ")
                print(f"@\t\t{string} | {guess} | {original} | {prediction}")

        score, aug_score = evaluate(data, _val_loader, encoder, decoder)

        print(f"@\t\tJaccard score: {score}\t Augmented Jaccard Score"
              f":{aug_score}")


# Validation cycle. Take the inputs and measure jaccard accuracy.
def evaluate(data, _val_loader, encoder, decoder, max_length=100):
    encoder.eval()
    decoder.eval()

    score = 0.
    aug_score = 0.

    print("Evaluating")
    for i, (x, y) in enumerate(_val_loader):
        x = x.to(device)
        y = y.to(device)

        encoder_hidden = encoder.init_hidden()

        x = x[0]
        y = y[0]

        input_length = x.size(0)
        target_length = y.size(0)

        prediction = ''
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                      device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                x[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[1]], device=device)
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            prediction += data.int_to_token[decoder_output.max(1)[1].item()] \
                          + " "
            if decoder_input.item() == "</s>":
                break
        target = tensor_to_words(y, data)
        original = tensor_to_words(x, data)
        _prediction = process_prediction(original, prediction)
        score += jaccard(target, prediction)
        aug_score += jaccard(target, _prediction)

    return score / len(val_loader), aug_score / len(val_loader)


# Accuracy metric for evaluation
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    div = len(a) + len(b) - len(c)
    if div == 0.:
        return 0.
    return float(len(c)) / div


# Utility functions used for converting from tensor to prediction.
def process_prediction(original: str, prediction: str):
    start = 100
    end = 0
    original = original.split()
    tokens = {}
    for token in prediction.split():
        tokens[token] = 1
    for i, token in enumerate(original):
        if token in tokens:
            start = min(i, start)
            end = max(i, end)
    return " ".join(original[start:end + 1])


def tensor_to_words(tensor, data):
    words = ''
    for ti in tensor:
        words += data.int_to_token[ti.item()] + " "
    return words


if __name__ == '__main__':
    print(f"#\tTorch Version: {torch.__version__}")
    # Attempt to grab CUDA. Grab cpu if that fails
    use_cuda = args.cuda and torch.cuda.is_available()
    print(f"#\tCuda enabled: " + str(use_cuda))
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.torch_seed)

    # instantiate data set
    print(f"#\tLoading Data")
    train_data = Language("../data/kaggle/")

    test = Language("../data/kaggle/", False)
    train_idx, val_idx = train_data.split_data(args.tp, shuffle=True)

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_data, sampler=train_sampler)
    val_loader = DataLoader(train_data, sampler=val_sampler)

    hidden_size = 256

    vocab_size = len(train_data.counts)
    encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
    attn_decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
    # print(train_data.token_to_int["<s>"])
    train(train_data, train_loader, val_loader, encoder1, attn_decoder1, 5)
