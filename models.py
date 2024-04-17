import random
import pandas as pd

import torch
import torch.nn as nn
torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.h_2_o = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.softmax = nn.Softmax(dim=1)

        self.train_losses = []
        self.train_accuracies = []

        self.test_losses = []
        self.test_accuracies = []

        random.seed(0)

    def forward(self, input_tensor):
        embeds = self.embedding(input_tensor)
        hidden, _ = self.lstm(embeds)
        hidden = self.dropout(hidden)
        output = self.h_2_o(hidden.view(len(hidden), -1)[-1])
        output = self.log_softmax(output.view(1, -1))
        return output

    def get_all_hidden_states(self, input_tensor):
        embeds = self.embedding(input_tensor)
        hidden, _ = self.lstm(embeds)
        return hidden

    def get_all_hidden_and_outputs(self, input_tensor): # outputs as probabilities
        hidden = self.get_all_hidden_states(input_tensor)
        output = self.h_2_o(hidden.view(len(input_tensor), -1))
        output = self.softmax(output)
        return hidden, output

    def shuffle(self, X, y):
        temp = list(zip(X, y))
        random.shuffle(temp)
        return zip(*temp)

    def predict_class(self, input_tensor): # requires self.eval()
        # single input class index prediction
        output = self(input_tensor)
        return output.topk(1)[1].item()

    def predict_classes(self, X): # requires self.eval()
        # multiple input class index predictions
        with torch.no_grad():
            y_hat = [self.predict_class(input_tensor) for input_tensor in X]
        return y_hat
    
    def accuracy(self, X, y):
        self.eval()
        with torch.no_grad():
            outputs = torch.cat([self(input_tensor) for input_tensor in X])
            predictions = torch.cat([output.topk(1)[1] for output in outputs])

            self.class_prediction_counts = pd.Series(predictions).value_counts().sort_index()
            
            n_correct = torch.sum(predictions == torch.tensor(y)).item()
        acc = n_correct/len(X)     
        return acc

    def train_instance(self, input_tensor, category_tensor, loss_criterion, optimizer):
        # training on a single input
        self.zero_grad()
        output = self(input_tensor)

        loss = loss_criterion(output, category_tensor)
        loss.backward()
        optimizer.step()

    def train_epoch(self, X_train, y_train, loss_criterion, optimizer):
        # applying self.train_instance to each training instance
        self.train()
        for i in range(len(X_train)):
            self.train_instance(X_train[i], y_train[i], loss_criterion, optimizer)
                        
    def fit(self, X_train, y_train, optimizer, loss_criterion,
            epochs=1,
            track_train_stats=False,
            track_test_stats=False,            
            X_test=None, y_test=None,
            test_every=1,
            verbose=False):

        if track_test_stats and (X_test is None or y_test is None):
            raise ValueError('Please specify X and y test sets.')
        
        for epoch in range(epochs):
            if verbose:
                print('Epoch ' + str(epoch+1))
                
            X_train_epoch, y_train_epoch = self.shuffle(X_train, y_train)
            self.train_epoch(X_train_epoch, y_train_epoch, loss_criterion, optimizer)

            if track_train_stats: # !!!               
                if epoch % test_every == 0:
                    train_accuracy = self.accuracy(X_train, y_train)
                    self.train_accuracies.append(train_accuracy)
                    if verbose:
                        print('Train Accuracy: ' + str(round(train_accuracy, 4)))
                                                   
            if track_test_stats:
                if epoch % test_every == 0:
                    test_accuracy = self.accuracy(X_test, y_test)
                    self.test_accuracies.append(test_accuracy)
                    if verbose:
                        print('Test Accuracy: ' + str(round(test_accuracy, 4)))
                    
        
