"""
This module contains helper functions that colorize data in various ways for
different visualizations in the dashboard.

Author: Steve Bischoff
"""
from dash import html

# Define globals
INT_COLOR_MAP = {0: [0,63,92], 1: [188,80,144], 2: [255,166,0]}
STRING_COLOR_MAP = {0: 'rgba(0,63,92,', 1: 'rgba(188,80,144,', 2: 'rgba(255,166,0,'}


def get_genre_key_text(genres):
    """
    RGBA colorizes genre name text using INT_COLOR_MAP to create a color key.
    Used in center column -> key text.
    Params:
        genres - list of genre strings
    Returns:
        genre_key_text - colorized html text
    """
    genre_key_text = []
    for i, genre in enumerate(genres):
        r, g, b = INT_COLOR_MAP[i]
        rgba_color = f'rgba({r}, {g}, {b}, 0.7)' # !!! Just use STRING_COLOR_MAP[i]+'0.7)'???
        genre_key_text.append(html.Span(genres[i],
                                        style={'background-color': rgba_color,
                                               'font-size': 14}))
        genre_key_text.append(html.Span(" ", style={'font-size': 14}))  # Add space between words
    return genre_key_text


def get_highlighted_description_text(words, fetched, max_alpha=0.7): # left -> Current Sequence
    """
    RGBA colorizes description text with rgb values determined by predicted
    class and alpha determined by confidence.
    Used in left column -> Current Sequence.
    Params:
        words - list of description text
        fetched - entire query result
    Returns:
        highlighted_text - colorized html text
    """
    highlighted_text = []
    for (word, out) in zip(words, fetched):
        prediction = out[6]
        alpha = out[7]*max_alpha
        r, g, b = INT_COLOR_MAP[prediction]
        rgba_color = f'rgba({r}, {g}, {b}, {alpha})'
        highlighted_text.append(html.Span(word,
                                          style={'background-color': rgba_color,
                                                 'font-size': 12}))
        highlighted_text.append(html.Span(" ", style={'font-size': 12}))  # Add space between words
    return highlighted_text


def predictions_to_string_colors(predictions, confidences, max_alpha=1):
    """
    Takes predictions and confidences to RGBA color strings.
    Used in left column -> Hidden State Distances, middle column -> Projection.
    Params:
        predictions - list of int class predictions
        confidences - list of floats in [0, 1]
        max_alpha - float in [0, 1]
    Returns:
        color_scale - list of RGBA color strings
    """
    color_scale = [STRING_COLOR_MAP[pred]+str(round(conf*max_alpha, 3))+')' for (pred, conf) in zip(predictions, confidences)]
    return color_scale


def get_heatmap_colors(outputs, max_alpha=255):
    """
    Takes 2D confidence outputs to 2D list of RGBA integers.
    Used in left column -> Class Contribution
    Params:
        outputs - list of list of floats in [0, 1] denoting class confidences
        max_alpha - float in [0, 255]
    Returns:
        list of RGBA integer lists
    """
    colors1 = []
    colors2 = []
    colors3 = []
    line_colors = []
    for output in outputs:
        confidence1 = output[0].item()
        colors1.append(INT_COLOR_MAP[0] + [int(confidence1*max_alpha)])

        confidence2 = output[1].item()
        colors2.append(INT_COLOR_MAP[1] + [int(confidence2*max_alpha)])
        
        confidence3 = output[2].item()
        colors3.append(INT_COLOR_MAP[2] + [int(confidence3*max_alpha)])
    return [colors1, colors2, colors3]
