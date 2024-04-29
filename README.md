# NLP-Genre-Classification

Data Source: https://www.kaggle.com/datasets/nelepie/imdb-genre-classification

Paper Cited: https://vciba.springeropen.com/articles/10.1186/s42492-021-00090-0

My Dashboard: https://nlp-genre-classification-68136a2a1ad7.herokuapp.com/


In this package, I create an LSTM model in PyTorch to predict a movie's genre based on its IMDB description. I then recreate the model visualization techniques from Garcia *et. al* 2021. I use Dash and Heroku to create an interactive dashboard. The user can interact with the app by clicking on rows in the Movies table to show information about the model's reasoning for that example. All examples are from the test set.

The Jupyter notebook in IMDB_genres_description_words.ipynb contains a detailed walkthrough of my process, as well as an explanation of each visualization in the dashboard.

Python code and CSS assets for the app can be found in the heroku_app folder in this repository. The app connects to cloud data using Google Cloud clients and performs database queries using SQLAlchemy.
