# NLP-Genre-Classification

My Dashboard: https://nlp-genre-classification-68136a2a1ad7.herokuapp.com/

Paper Cited: https://vciba.springeropen.com/articles/10.1186/s42492-021-00090-0

Data Source: [https://www.kaggle.com/datasets/nelepie/imdb-genre-classification](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

## Overview

In this package, I create LSTM and Transformer models in PyTorch to predict a movie's genre based on its IMDB description. I then recreate the LSTM visualization techniques from Garcia *et. al* 2021. I use Dash and Heroku to create an interactive dashboard. The user can interact with the app by clicking on rows in the Movies table to show information about the LSTM model's reasoning for that example. All examples are from the test set.

## Files

 * **IMDB_genre_classification.ipynb**: this Jupyter notebook contains a detailed walkthrough of my process, as well as an explanation of each visualization in the dashboard.
 * **imdb_data_prep.py**: this script takes the raw description files from Kaggle and preprocesses them to be read into PyTorch models.
 * **imdb_dataset.py**: this file contains a PyTorch Dataset subclass to create custom dataset objects.
 * **models.py**: this file contains the model classes (subclasses of torch.nn.Module)
 * **heroku_app**: Python code and CSS assets for the app can be found in the heroku_app folder in this repository. The app connects to cloud data using Google Cloud clients and performs database queries using SQLAlchemy.
