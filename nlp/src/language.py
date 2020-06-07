from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from collections import Counter
from string import punctuation


class Language(Dataset):
    def __init__(self, path: str, train: bool = True):
        super(Language).__init__()
        self.train = train

        self.data = self.load_data(path)
        self.data_length = len(self.data.index - 1)

        self.line_length = 0
        self.word_length = 0
        self.token_to_int = None
        self.int_to_token = None

        self.counts = Counter({"<UNK>": 1})

        self.vocab = {'<UNK>': 0}
        # convert_labels(self.data, lookup_dict)
        self.clean_data(self.data)
        self.encode_dict()
        self.encode_data()

    def __getitem__(self, item):
        if item >= self.data_length:
            raise StopIteration

        return torch.LongTensor(self.data.at[item, 'text']), \
               torch.LongTensor(self.data.at[item, 'selected_text'])

    def __len__(self):
        return self.data_length

    def load_data(self, path: str):
        df = pd.read_csv(path + ("train.csv" if self.train else "test.csv"))
        return df

    def split_data(self, t_split: float, shuffle=True,
                   seed: int = 42) -> (np.ndarray, np.ndarray):
        idx = list(range(self.data_length))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(idx)

        # XXX_idx is the upper bound for that range
        val_idx = int(self.data_length * t_split)

        return idx[:val_idx], idx[val_idx:]

    # Run on single line so that it can be used during test
    def clean_data(self, data: pd.DataFrame):

        for i in data.index:
            text = data.at[i, 'text']
            if self.train: target = data.at[i, 'selected_text']
            sentiment = data.at[i, 'sentiment']

            if type(text) != str:
                text = ""
                target = ""
            # get rid of punctuation and append sentiment + start and end tokens
            text = text.lower()
            text = ''.join([c for c in text if c not in punctuation])
            text = f"{sentiment} {text}"
            text = text.split()
            if self.train:
                target = target.lower()
                target = ''.join([c for c in target if c not in punctuation])
                target = f"{target}"
                target = target.split()

            # count occurrences TODO: implement GloVe
            self.counts.update(text)
            self.line_length = max(len(text), self.line_length)
            data.at[i, 'text'] = text
            if self.train: data.at[i, 'selected_text'] = target

    # FIXME: This self.train pipeline is ugly and non-cohesive.
    def encode_data(self):
        for i in self.data.index:
            text = self.data.at[i, 'text']
            if self.train: target = self.data.at[i, 'selected_text']

            if type(text) != list:
                continue

            self.data.at[i, 'text'] = \
                [self.token_to_int[token] for token in text]
            selected_text = []
            if self.train:
                for token in target:
                    selected_text.append(self.token_to_int[token]
                                         if token in self.token_to_int else
                                         self.token_to_int["<UNK>"])
                self.data.at[i, 'selected_text'] = selected_text

    def encode_dict(self):
        tokens = sorted(self.counts, key=self.counts.get, reverse=True)

        self.token_to_int = {word: idx + 1 for idx, word in enumerate(tokens)}
        self.int_to_token = {idx: word for word, idx in
                             self.token_to_int.items()}
