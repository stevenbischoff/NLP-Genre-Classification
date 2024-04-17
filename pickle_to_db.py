import pickle
import sqlite3
import pandas as pd
import numpy as np
import torch

from models import *

genres3 = {0:'drama', 1:'documentary', 2:'comedy'}

with open('data/imdb_hiddens3.pkl', 'rb') as f:
    all_hiddens = pickle.load(f) # list of tensors

with open('data/imdb_outputs3.pkl', 'rb') as f:
    all_outputs = pickle.load(f)

with open('data/imdb_predictions3.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

layer_confidences = [[output.topk(1)[0].item() for output in outputs] for outputs in all_outputs]
final_confidences = [outputs[-1].topk(1)[0].item() for outputs in all_outputs]

df3 = pd.read_pickle('data/imdb_train_processed_3_genres.pkl')

####################################################
# Text Data
## Descriptions: movie_id, description, genre, predicted_genre, confidence
text_data = [(i,
              ' '.join(df3.loc[i, 'clean_description_list_trunc']),
              df3.loc[i, 'genre'],
              genres3[all_predictions[i][-1].item()],
              final_confidences[i]
              ) for i in range(len(df3))]
##########################
conn = sqlite3.connect('data/imdb_app.db')
cursor = conn.cursor()

cursor.execute('''DELETE FROM Descriptions''')
conn.commit()
conn.close()
##########################
conn = sqlite3.connect('data/imdb_app.db')
cursor = conn.cursor()

cursor.executemany('''INSERT INTO Descriptions(movie_id, description, genre, predicted_genre, confidence) VALUES(?, ?, ?, ?, ?)''',
                   text_data)
conn.commit()
conn.close()
##########################
####################################################
# Hiddens / Outputs data
## Hiddens_Outputs: movie_id, layer_id, hiddens, (!!!hiddens_2d!!!), outputs, prediction
ho_data = [(i,
            j,
            hidden.detach().numpy().tobytes(),
            all_outputs[i][j].detach().numpy().tobytes(),
            all_predictions[i][j].item(), layer_confidences[i][j]
            ) for i, hiddens in enumerate(all_hiddens) for j, hidden in enumerate(hiddens)]
##########################
conn = sqlite3.connect('data/imdb_app.db')
cursor = conn.cursor()

cursor.execute('''DELETE FROM Hiddens_Outputs''')
conn.commit()
conn.close()
##########################
conn = sqlite3.connect('data/imdb_app.db')
cursor = conn.cursor()

cursor.executemany('''
INSERT INTO Hiddens_Outputs(movie_id, layer_id, hiddens, outputs, prediction, confidence)
VALUES(?, ?, ?, ?, ?, ?)
''', ho_data)
conn.commit()
conn.close()
####################################################
##########################
conn = sqlite3.connect('data/imdb_app.db')
cursor = conn.cursor()

res = cursor.execute('''SELECT * FROM Hiddens_Outputs''')
i = res.fetchone()[3]
print(np.frombuffer(i, dtype=np.float32)) # need to specify dtype=np.float32 !!!

#print([np.frombuffer(text_byte_tuple[0]) for text_byte_tuple in res.fetchall()[-10:]])
conn.close()

# Current Sequence
## Description text
## Description colors (top)
# Class Contribution
## Description colors (all)
# Hidden State Distances
## Description hiddens
## Colors (top)

# Projection of Hidden States
## Sample of 2D hiddens with colors (top)
### Load full dataframe
## Description 2D hiddens with colors (top)

# Sequences
## All (?) description texts
## Actual genres
## Predicted genres

### DATABASE
## Description Table
# movie id
# text
# actual genre
## Hiddens / Outputs Table
# movie id (same as Description Table)
# word id (in [0, n], where 0 < n < 100)
# hiddens
# 2D hiddens
# outputs (to serve as alphas for colors)
# predicted genre after each word

