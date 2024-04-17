from models import *
from sklearn.decomposition import IncrementalPCA
import pickle

genres = ['drama', 'documentary', 'comedy']

# Load preprocessed datasets
X_train = torch.load('data/X_train{}.pt'.format(len(genres)))
y_train = torch.load('data/y_train{}.pt'.format(len(genres)))
X_test = torch.load('data/X_test{}.pt'.format(len(genres)))
y_test = torch.load('data/y_test{}.pt'.format(len(genres)))
vocab = torch.load('saved_models/vocab{}.pt'.format(len(genres)))

vocab_size = len(vocab)
embedding_size = 64
hidden_size = 128
output_size = len(genres)

# Load trained model
lstm = LSTM(vocab_size, embedding_size, hidden_size, output_size)
lstm.load_state_dict(torch.load('saved_models/imdb_words_100_60_lstm{}.pt'.format(len(genres))))

# Save model hidden states and outputs
all_hidden_states = []
all_outputs = []
all_predictions = []
for i, X in enumerate(X_train):    
    hidden, output = lstm.get_all_hidden_and_outputs(X)
    prediction = output.topk(1)[1]

    all_hidden_states.append(hidden)
    all_outputs.append(output)
    all_predictions.append(prediction)

hidden_states_tensor = torch.squeeze(torch.cat(all_hidden_states))

with open('data/imdb_hiddens{}.pkl'.format(len(genres)), 'wb') as f:
    pickle.dump(all_hidden_states, f)
del all_hidden_states

with open('data/imdb_outputs{}.pkl'.format(len(genres)), 'wb') as f:
    pickle.dump(all_outputs, f)
del all_outputs

with open('data/imdb_predictions{}.pkl'.format(len(genres)), 'wb') as f:
    pickle.dump(all_predictions, f)
del all_predictions

# Apply PCA to hidden states and save
ipca = IncrementalPCA(n_components=2, batch_size=100)
hidden_pca = ipca.fit_transform(hidden_states_tensor.detach().numpy())

with open('data/imdb_hiddens{}_pca.pkl'.format(len(genres)), 'wb') as f:
    pickle.dump(hidden_pca, f)

with open('saved_models/pca{}.pkl'.format(len(genres)), 'wb') as f:
    pickle.dump(ipca, f)
