import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
torch.manual_seed(0)

class pack_pad_lstm_wrapper(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, batch_size=1):
        super().__init__()

        self.padding_value = vocab['<pad>']

        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)

    def forward(self, embeds, lengths):
        # pack padded embeddings using lengths list for lstm layer
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.lstm(packed_embeds)
        # pad packed hidden layers using vocab padding value
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, padding_value=self.padding_value)

        return hidden
    

class LSTM(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, output_size, batch_size=1):
        super().__init__()

        self.embedding = nn.Embedding(len(vocab), embedding_size)
        self.pack_pad_lstm = pack_pad_lstm_wrapper(vocab, embedding_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.h_2_o = nn.Linear(hidden_size, output_size)
        
        # activations: log_softmax for training, softmax for getting probabilities
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx, lengths):
        embeds = self.embedding(idx)

        hidden = self.pack_pad_lstm(embeds, lengths)
         
        hidden = self.dropout(hidden)
        # keep only the last unpadded hidden state using lengths
        hidden = torch.stack([h[length-1] for h, length in zip(hidden, lengths)])
        output = self.h_2_o(hidden)
        output = self.log_softmax(output)
        
        return output

    @torch.no_grad()
    def evaluate_batch(self, idx, lengths, y, loss_criterion): # assumes self.eval()
        """
        Calculate loss and accuracy (# correct and # total) for a batch of inputs.
        Assumes self is in evaluation mode (self.eval()).
        Params:
            idx: tensor of shape (batch_size, max_length)
            lengths: list of integers
            y: tensor of shape (batch_size)
        Returns:
            loss: scalar
            n_correct: int
            n_total: int
        """
        outputs = self(idx, lengths)
        predictions = outputs.topk(1)[1]
        # calculate loss
        loss = loss_criterion(outputs, y).item()    
        # calculate no. of correct predictions              
        n_correct = torch.sum(predictions == y.view(-1, 1)).item() 

        return loss, (n_correct, len(idx))
    
    @torch.no_grad()
    def evaluate(self, dataloader, loss_criterion):
        """
        Calculate loss and accuracy for all inputs in a dataloader.
        Params:
            dataloader: DataLoader object for IMDBDataset
            loss_criterion: torch loss function
        Returns:
            loss: float
            acc: float
        """
        # initialize variables
        losses = []
        n_correct, n_total = 0, 0
        # set evaluation mode
        self.eval()
        # iterate over batches
        for i, (idx, lengths, y) in enumerate(dataloader): # iterate over batches
            batch_loss, (batch_correct, batch_size) = self.evaluate_batch(idx, lengths, y, loss_criterion)
            losses.append(batch_loss)
            n_correct += batch_correct
            n_total += batch_size
        acc = n_correct/n_total
        
        return np.mean(losses), acc

    def train_batch(self, idx, lengths, y, loss_criterion, optimizer):
        """
        Train the model on a single batch.
        Params:
            idx: tensor of shape (batch_size, max_length)
            lengths: list of integers
            y: tensor of shape (batch_size)
            loss_criterion: torch loss function
            optimizer: torch optimizer
        """
        # reset gradients
        self.zero_grad()
        # get batch model output
        output = self(idx, lengths)
        # calculate loss
        loss = loss_criterion(output, y)
        # backpropagate
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def train_epoch(self, dataloader, loss_criterion, optimizer):
        """
        Train the model for one epoch.
        Params:
            dataloader: DataLoader object for IMDBDataset
            loss_criterion: torch loss function
            optimizer: torch optimizer
        """
        # set training mode
        self.train() 
        # iterate over batches
        for i, (idx, lengths, y) in enumerate(dataloader):
            self.train_batch(idx, lengths, y, loss_criterion, optimizer)

    def fit(self, 
            train_dataloader,
            loss_criterion,
            optimizer, 
            epochs=1, # int >= 0
            track_train_stats=False, # bool
            track_test_stats=False, # bool
            test_dataloader=None, # DataLoader object for IMDBDataset
            verbose=False # bool
            ):
        """
        Train the model for a specified number of epochs.
        Params:
            train_dataloader: DataLoader object for IMDBDataset
            loss_criterion: torch loss function
            optimizer: torch optimizer
        """
        # sanity check
        assert not track_test_stats or test_dataloader is not None, 'Please set track_test_stats to False or specify test dataloader.'
        # iterate over epochs
        for epoch in range(epochs):
            if verbose:
                print('Epoch ' + str(epoch+1))
            # run one training epoch
            self.train_epoch(train_dataloader, loss_criterion, optimizer)
            # optionally track training stats
            if track_train_stats:
                train_loss, train_accuracy = self.evaluate(train_dataloader, loss_criterion)
                if verbose:
                    print('Train Loss: {0:.3f}. Train Accuracy: {1:.3f}'.format(train_loss, train_accuracy))
            # optionally track test stats
            if track_test_stats:
                test_loss, test_accuracy = self.evaluate(test_dataloader, loss_criterion)
                if verbose:
                    print('Test Loss:  {0:.3f}. Test Accuracy:  {1:.3f}'.format(test_loss, test_accuracy))
