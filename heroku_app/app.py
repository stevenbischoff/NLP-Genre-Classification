from dash import Dash, html, dcc, callback, Output, Input, dash_table
from flask_caching import Cache
from google.cloud import storage 
from google.cloud.sql.connector import Connector
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlalchemy

import color_functions as cf

# Define globals
GENRES = {0:'drama', 1:'documentary', 2:'comedy'}
from gcloud_credentials import *

# Initialize app
app = Dash(__name__)
server = app.server

# Initialize cache for 2D projection scatterplot
cache = Cache(server, config={'CACHE_TYPE': 'SimpleCache'})

##### GCLOUD #####
# load 2D projection sample data
storage_client = storage.Client(project='bright-aileron-421120')
bucket = storage_client.bucket('bright-aileron-421120.appspot.com')
blob = bucket.blob('imdb_hiddens{}_pca_sample_test.pkl'.format(len(GENRES)))
pickle_in = blob.download_as_string()
hidden_pca_sample = pickle.loads(pickle_in)
# Load movie dataframe
blob = bucket.blob('imdb_test_processed_{}_genres_app.pkl'.format(len(GENRES)))
pickle_in = blob.download_as_string()
table_df = pickle.loads(pickle_in)
# Initialize database connector
connector = Connector()
# function to return the database connection object
def getconn():
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        'pymysql',
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME
    )
    return conn
# Create connection pool with 'creator' argument to our connection object function
pool = sqlalchemy.create_engine('mysql+pymysql://', creator=getconn)
##### END GCLOUD #####

##### APP #####
app.layout = html.Div([
    # Left column
    html.Div([
        html.Div(
            children=[
                html.H4('Sequence Overview', className='component-header'),
                dash_table.DataTable(
                    id='overview-table',
                    style_cell={
                        'fontSize': 12,
                        'fontFamily': 'Times New Roman',
                        'backgroundColor': 'white',
                        }
                    )
                ],
            className='left-component-container',
            style={'height':'24vh'}
            ),      
        html.Div(
            children=[
                html.H4('Current Sequence', className='component-header'),
                html.Div(id='current-sequence', className='component-element')
                ],
            className='left-component-container',
            style={'height':'23vh'}
            ),
        html.Div(
            children=[
                html.H4('Class Contribution', className='component-header'),
                dcc.Graph(id='class-contribution', style={'margin': '0px'})
                ],
            className='left-component-container',
            style={'height':'16.5vh'}
            ),
        html.Div(
            children=[
                html.H4('Hidden State Distances', className='component-header'),
                dcc.Graph(id='distances', className='component-element')
                ],
            className='left-component-container',
            style={'height':'33.5vh'}
            ),
        ], className='left-column'
    ),
    # Middle column
    html.Div([
        html.H4('Projection of Hidden States of LSTM Layer', className='component-header'),
        dcc.Graph(id='projection', className='component-element'),
        html.H4('Color Key', className='component-header'),
        html.Div(cf.get_genre_key_text(GENRES), className='component-element')
        ], className='middle-column',
    ),
    # Right column
    html.Div([
        html.H4('Movies (click to select)', className='component-header'),
        html.Div([
            dash_table.DataTable(
                id='movie-datatable',
                data=table_df.to_dict('records'),
                page_size=20,
                columns=[{'name': i, 'id': i} for i in table_df.columns],
                style_cell={
                    'textAlign': 'left',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'fontSize': 12,
                    'fontFamily': 'Times New Roman',
                    },
                )              
            ], className='component-element')
        ], className='right-column',
    )
], style={'height':'97vh', 'padding':'0px', 'overflow':'hidden'})
##### END APP #####

##### CALLBACKS #####
@cache.memoize(timeout=120)
def get_base_projection_fig():
    # Get sample point colors
    scatter_colors = cf.predictions_to_string_colors(
        hidden_pca_sample['prediction'],
        hidden_pca_sample['confidence'],
        max_alpha=0.4 # 0.4
        )
    # Create figure
    projection_fig = make_subplots()
    projection_fig.add_trace(
        go.Scatter(
            x=hidden_pca_sample[0],
            y=hidden_pca_sample[1],
            mode='markers',
            marker={'color': scatter_colors},
            )
        )
    return projection_fig

@app.callback(
    [Output('overview-table', 'data'),
     Output('current-sequence', 'children'),
     Output('class-contribution', 'figure'),
     Output('distances', 'figure'),
     Output('projection', 'figure')],
    [Input('movie-datatable', 'active_cell'),
     Input('movie-datatable', 'page_current'),
     Input('movie-datatable', 'page_size')]
)
def update_components(active_cell, page_current, page_size):
    ### Calculate movie_id from row and current page    
    row = active_cell['row'] if active_cell else 0
    page_current = page_current if page_current else 0
    movie_id = page_current*page_size + row
    
    ### Fetch data using movie_id 
    with pool.connect() as db_conn:
        fetched = db_conn.execute(sqlalchemy.text(
            "SELECT * FROM Hiddens_Outputs_Test " # !!!
            "WHERE movie_id = {}".format(movie_id)
            )).fetchall()
    
        fetched_desc = db_conn.execute(sqlalchemy.text(
            "SELECT * FROM Descriptions_Test " # !!!
            "WHERE movie_id = {}".format(movie_id)
            )).fetchone()

    ### Process data
    ## Descriptions
    words = fetched_desc[2].split()
    movie_class = fetched_desc[3]
    predicted_class = fetched_desc[4]
    correct = movie_class == predicted_class
    confidence = fetched_desc[5]

    ## h_o
    outputs = [np.frombuffer(h_o[5], dtype=np.float32) for h_o in fetched] # 4 local
    distances_norm = [h_o[3]for h_o in fetched] # 2 local 
    predictions = [h_o[6] for h_o in fetched] # Used in Projection.
    confidences = [h_o[7] for h_o in fetched] # Used in Projection.
    hidden_pca = [np.frombuffer(h_o[4], dtype=np.float32) for h_o in fetched] # Used only in Projection.
    hidden_pca_df = pd.DataFrame.from_records(hidden_pca)

    ### Create figures
    ## Sequence Overview
    table_data = pd.DataFrame({'Sequence': ['Class', 'Classification', 'Correctly Classified', 'Confidence'],
                         str(movie_id): [movie_class, predicted_class, correct, round(confidence, 3)]}
                        ).to_dict('records')

    ## Current Sequence    
    highlighted_text = cf.get_highlighted_description_text(words, fetched)

    ## Class Contribution    
    heatmap_colors = np.array(cf.get_heatmap_colors(outputs), dtype=np.uint8)
    
    cc_fig = px.imshow(
        heatmap_colors,
        y=[0,8,16],
        )
    cc_fig.update_yaxes(
        automargin='left',
        tickmode='array',
        tickvals=[0,8,16],
        ticktext=['drama', 'documentary', 'comedy'],
        )
    
    cc_fig.update_layout(
        margin={'b':0, 'l':0, 'r':0, 't':0},
        height=100,
        plot_bgcolor='white',
        font_family='Times New Roman',
        )
    
    cc_fig.update(data=[{'customdata': np.dstack((np.array([[f'{i: .2f}' for i in output] for output in outputs]).T,
                                                  [words, words, words])),
        'hovertemplate': "word=%{customdata[1]}<br>confidence=%{customdata[0]}"}])
    

    ## Distances
    movie_colors = cf.predictions_to_string_colors(predictions, confidences, max_alpha=1)
    distance_df = pd.DataFrame({'i': [i for i in range(len(distances_norm))],
                                'distance': distances_norm,
                                'color': movie_colors,
                                'word': words})

    distance_fig = px.bar(distance_df,
                          x='i', y='distance', color='color',
                          color_discrete_map = 'identity',
                          hover_data={'i':False, 'word':True, 'distance':':.3f'})
    
    distance_fig.update_xaxes(visible=False)
    distance_fig.update_yaxes(
        visible=False,
        range=[0,1],
        automargin='left')
    distance_fig.update_layout(
        showlegend=False,
        margin={'b':0, 'l':5, 'r':5, 't':10},
        height=200,
        plot_bgcolor='white'
        )
    
    ## Projection
    # scatter
    projection_fig = get_base_projection_fig()

    # lines   
    # marker for first word
    projection_fig.add_trace(
        go.Scatter(
            x=[hidden_pca_df.iloc[0, 0]],
            y=[hidden_pca_df.iloc[0, 1]],
            mode='markers',
            marker={'color': movie_colors[0], 'size':10, 'line':{'width':1.5}},
            )
        )
    # lines for subsequent words
    for i in range(len(hidden_pca_df)-1):       
        projection_fig.add_trace(
            go.Scatter(
                x=hidden_pca_df.iloc[i:i+2, 0],
                y=hidden_pca_df.iloc[i:i+2, 1],
                mode='lines',
                line={'color':movie_colors[i+1], 'width':3}
                )
            )

    # layout
    projection_fig.update_xaxes(visible=False)
    projection_fig.update_yaxes(visible=False)
    projection_fig.update_layout(
        showlegend=False,
        margin={'b':0, 'l':5, 'r':5, 't':10},
        plot_bgcolor='white'
        )
    
    return table_data, highlighted_text, cc_fig, distance_fig, projection_fig
##### END CALLBACKS #####

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=7090)

    connector.close()
