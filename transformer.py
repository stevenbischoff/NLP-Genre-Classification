"""
This module contains a custom PyTorch encoder-only Transformer class, built up
from several sub-modules.

modified from: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
"""
import numpy as np

import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, output_size)
                
        self.log_softmax = nn.LogSoftmax(dim=1)     
        self.softmax = nn.Softmax(dim=-1)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        x = self.lm_head(x)[:,0,:] # (B,output_size)  
        
        x = self.log_softmax(x)
        return x

    def evaluate_batch(self, idx, y, loss_criterion): # assumes self.eval()
        """
        Calculate loss and accuracy (# correct and # total) for a batch of inputs.
        Assumes self is in evaluation mode (self.eval()).
        Params:
            idx: tensor of shape (batch_size, max_length)
            y: tensor of shape (batch_size)
            loss_criterion: torch loss function
        Returns:
            loss: scalar
            n_correct: int
            n_total: int
        """
        outputs = self(idx)
        predictions = outputs.topk(1)[1]
        # calculate loss
        loss = loss_criterion(outputs, y).item()    
        # calculate no. of correct predictions              
        n_correct = torch.sum(predictions == y.view(-1, 1)).item() 

        return loss, (n_correct, len(idx))
    
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
        with torch.no_grad():
            for i, (idx, lengths, y) in enumerate(dataloader): # iterate over batches
                batch_loss, (batch_correct, batch_size) = self.evaluate_batch(idx, y, loss_criterion)
                losses.append(batch_loss)
                n_correct += batch_correct
                n_total += batch_size
        acc = n_correct/n_total
        return np.mean(losses), acc

    def train_batch(self, idx, y, loss_criterion, optimizer):
        """
        Train the model on a single batch.
        Params:
            idx: tensor of shape (batch_size, max_length)
            y: tensor of shape (batch_size)
            loss_criterion: torch loss function
            optimizer: torch optimizer
        """
        # reset gradients
        self.zero_grad()
        # get batch model output
        output = self(idx)
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
            self.train_batch(idx, y, loss_criterion, optimizer)

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
            loss_criterion: torch loss function
            optimizer: torch optimizer
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
                    if verbose:
                        print('Train Loss: ' + str(round(train_loss, 3)) + '. Train Accuracy: ' + str(round(train_accuracy, 3)))
            # optionally track test stats
            if track_test_stats:
                if epoch % test_every == 0:
                    test_loss, test_accuracy = self.evaluate(test_dataloader, loss_criterion)
                    if verbose:
                        print('Test Loss:  ' + str(round(test_loss, 3)) + '. Test Accuracy:  ' + str(round(test_accuracy, 3)))

# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T) 
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)

        return x
