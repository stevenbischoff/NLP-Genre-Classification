import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, output_size, batch_size=1):
        super().__init__()

        self.padding_value = vocab['<pad>']
        self.vocab_size = len(vocab)

        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.h_2_o = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.softmax = nn.Softmax(dim=-1)

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def forward(self, input_tensor, lengths):
        embeds = self.embedding(input_tensor)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.lstm(packed_embeds)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, padding_value=self.padding_value)
        hidden = self.dropout(hidden)
        # keep only the last unpadded hidden state using lengths
        hidden = torch.stack([h[length-1] for h, length in zip(hidden, lengths)])
        output = self.h_2_o(hidden)
        output = self.log_softmax(output)
        return output
    
    def get_batch_hiddens_and_outputs(self, input_tensor, lengths, mode='numpy'):
        """
        Get unpadded hidden states and outputs for a batch of inputs.
        Returns lists of tensors because unpadded hiddens and outputs vary in length.
        Params:
            input_tensor: tensor of shape (batch_size, max_length)
            lengths: list of integers
            mode: 'numpy' or 'torch'
        Returns:
            hidden: list of tensors of varying shapes (time_steps, hidden_size)
            output: list of tensors of varying shapes (time_steps, hidden_size)
        """
        embeds = self.embedding(input_tensor)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.lstm(packed_embeds)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, padding_value=self.padding_value)
        output = self.h_2_o(hidden)
        output = self.softmax(output)

        if mode == 'numpy':
            hidden = [h[:length].detach().numpy() for h, length in zip(hidden, lengths)]
            output = [o[:length].detach().numpy() for o, length in zip(output, lengths)]
        elif mode == 'torch':
            hidden = [h[:length] for h, length in zip(hidden, lengths)]
            output = [o[:length] for o, length in zip(output, lengths)]

        return hidden, output
    
    def get_all_hiddens_and_outputs(self, dataloader, mode='numpy'):
        """
        Get all hidden states and outputs for a dataloader.
        Params:
            dataloader: DataLoader object for IMDBDataset
            mode: 'numpy' or 'torch'
        Returns:
            all_hiddens: list of tensors of varying shapes (time_steps, hidden_size)
            all_outputs: list of tensors of varying shapes (time_steps, output_size)
        """
        # initialize variables
        all_hiddens, all_outputs = [], []
        # set evaluation mode
        self.eval()
        with torch.no_grad():
            for i, (input_tensor, lengths, category_tensor) in enumerate(dataloader):
                batch_hiddens, batch_outputs = self.get_batch_hiddens_and_outputs(input_tensor, lengths, mode=mode)
                all_hiddens += batch_hiddens
                all_outputs += batch_outputs
        return all_hiddens, all_outputs

    def evaluate_batch(self, X, lengths, y, loss_criterion): # assumes self.eval()
        """
        Calculate loss and accuracy (# correct and # total) for a batch of inputs.
        Assumes self is in evaluation mode (self.eval()).
        Params:
            X: tensor of shape (batch_size, max_length)
            lengths: list of integers
            y: tensor of shape (batch_size)
        Returns:
            loss: scalar
            n_correct: int
            n_total: int
        """
        outputs = self(X, lengths)
        predictions = outputs.topk(1)[1]
        # calculate loss
        loss = loss_criterion(outputs, y).item()    
        # calculate no. of correct predictions              
        n_correct = torch.sum(predictions == y.view(-1, 1)).item() 

        return loss, (n_correct, len(X))
    
    def evaluate(self, dataloader, loss_criterion):
        """
        Calculate loss and accuracy for all inputs in a dataloader.
        Params:
            dataloader: DataLoader object for IMDBDataset
            loss_criterion: loss function
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
        with torch.no_grad():
            for i, (input_tensor, lengths, category_tensor) in enumerate(dataloader): # iterate over batches
                batch_loss, (batch_correct, batch_size) = self.evaluate_batch(input_tensor, lengths, category_tensor, loss_criterion)
                losses.append(batch_loss)
                n_correct += batch_correct
                n_total += batch_size
        acc = n_correct/n_total
        return np.mean(losses), acc

    def train_batch(self, input_tensor, lengths, category_tensor, loss_criterion, optimizer):
        """
        Train the model on a single batch.
        Params:
            input_tensor: tensor of shape (batch_size, max_length)
            lengths: list of integers
            category_tensor: tensor of shape (batch_size)
            loss_criterion: loss function
            optimizer: optimizer
        """
        # reset gradients
        self.zero_grad()
        # get batch model output
        output = self(input_tensor, lengths)
        # calculate loss
        loss = loss_criterion(output, category_tensor)
        # backpropagate
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_epoch(self, dataloader, loss_criterion, optimizer):
        """
        Train the model for one epoch.
        Params:
            dataloader: DataLoader object for IMDBDataset
            loss_criterion: loss function
            optimizer: optimizer
        """
        # set training mode
        self.train() 
        # iterate over batches
        for i, (input_tensor, lengths, category_tensor) in enumerate(dataloader):
            batch_loss = self.train_batch(input_tensor, lengths, category_tensor, loss_criterion, optimizer)

    def fit(self, 
            train_dataloader,
            loss_criterion,
            optimizer, 
            epochs=1,
            track_train_stats=False, # bool
            track_test_stats=False, # bool
            test_dataloader=None, # DataLoader object for IMDBDataset
            test_every=1, # int >= 1
            verbose=False # bool
            ):
        """
        Train the model for a specified number of epochs.
        Params:
            train_dataloader: DataLoader object for IMDBDataset
            loss_criterion: loss function
            optimizer: optimizer
            epochs: int
        """
        # sanity checks
        assert test_every >= 1, 'test_every must be >= 1'
        assert not track_test_stats or test_dataloader is not None, 'Please set track_test_stats to False or specify test dataloader.'
        # iterate over epochs
        for epoch in range(epochs):
            if verbose:
                print('Epoch ' + str(epoch+1))
            # run one training epoch
            self.train_epoch(train_dataloader, loss_criterion, optimizer)
            # optionally track training stats
            if track_train_stats:
                if epoch % test_every == 0:
                    train_loss, train_accuracy = self.evaluate(train_dataloader, loss_criterion)
                    self.train_losses.append(train_loss)
                    self.train_accuracies.append(train_accuracy)
                    if verbose:
                        print('Train Loss: ' + str(round(train_loss, 3)) + '. Train Accuracy: ' + str(round(train_accuracy, 3)))
            # optionally track test stats
            if track_test_stats:
                if epoch % test_every == 0:
                    test_loss, test_accuracy = self.evaluate(test_dataloader, loss_criterion)
                    self.test_losses.append(test_loss)
                    self.test_accuracies.append(test_accuracy)
                    if verbose:
                        print('Test Loss:  ' + str(round(test_loss, 3)) + '. Test Accuracy:  ' + str(round(test_accuracy, 3)))
