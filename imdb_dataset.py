"""
This module contains a PyTorch Dataset subclass that gets fed into a PyTorch
Dataloader for model use.
  - X: padded, tokenized descriptions (same length)
  - lengths: lengths of the non-padded descriptions
  - y: genre token

Author: Steve Bischoff
"""
from torch.utils.data import Dataset

class IMDB_Dataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = X
        self.lengths = lengths
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.y[idx]
