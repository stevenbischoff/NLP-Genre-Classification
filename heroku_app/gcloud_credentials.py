import os

INSTANCE_CONNECTION_NAME = f'bright-aileron-421120:us-south1:nlp-analysis'
DB_USER = os.getenv('DB_USER') # change for local deployment
DB_PASS = os.getenv('DB_PASS') # change for local deployment
DB_NAME = 'imdb_app'
