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
