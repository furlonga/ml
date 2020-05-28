from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from string import punctuation


def load_data(path: str, _train: bool):
    df = pd.read_csv(path + ("train.csv" if _train else "test.csv"))
    return df


def convert_labels(_data: pd.DataFrame, conversion: dict):
    for i in _data.index:
        _data.at[i, "sentiment"] = conversion[_data.at[i, "sentiment"]]


class Data(Dataset):
    def __init__(self, path: str, lookup_dict: dict, train: bool = True):
        super(Data).__init__()

        self.data = load_data(path, train)
        convert_labels(self.data, lookup_dict)

        self.data_length = len(self.data.index - 1)
        self.line_length = 0
        self.vocab = {'<UNK>': 0}

    def __getitem__(self, item):
        if item >= self.data_length:
            raise StopIteration
        return self.data.at[item, 'text'], \
                self.data.at[item, 'selected_text'], \
                self.data.at[item, 'sentiment'],

    def __len__(self):
        return self.data_length

    def split_data(self, t_split: float, shuffle=True,
                   seed: int = 12) -> (np.ndarray, np.ndarray):
        """
            Shuffle indices of data set
        :param seed:  Seed for random shuffle
        :param t_split: Decimal representing percentage of data for training.
        :param shuffle: boolean to shuffle dataset or not.
        :return: Two lists of indices in dataset. First is train indices,
                 second is tests indices.
        """
        idx = list(range(self.data_length))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(idx)

        # XXX_idx is the upper bound for that range
        val_idx = int(self.data_length * t_split)

        return idx[:val_idx], idx[val_idx:]

    def clean_string(self, text: str):
        # get rid of punctuation
        text = text.lower()
        text = ''.join([c for c in text if c not in punctuation])
        text = text.split()

        # count occurences
        for word in text:
            if word not in self.vocab:
                self.vocab[word] = 0
            self.vocab[word] += 1

        self.line_length = max(len(text), self.line_length)

        return text

