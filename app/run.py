# import libraries
import sys
import time
from collections import Counter
import pandas as pd
import numpy as np
import json
import plotly
from plotly.graph_objs import Bar

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import joblib
from sqlalchemy import create_engine
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from pprint import pprint
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')


app = Flask(__name__)

def tokenize(text):
    """
    A function that accepts a text string and tokenizes the same
    Specifically, it performs the following operations on the text:
    * Normalize text by removing anything other than alphabets and numbers
      and convert to lower case
    * Tokenize text into a list of words
    * Remove stopwords using the "english" option
    * Extract word roots for the words in the word list
    :param text: (str) source string
    :return: (list) cleaned words
    """

    # Normalize text by removing anything other than alphabets and numbers
    # and convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text into a list of words
    words = word_tokenize(text)

    # Remove stopwords using the "english" option
    words = [w for w in words if w not in stopwords.words("english")]

    # Extract word roots for the words in the word list
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Plotting Categories
    categories_pct = df[df.columns[4:]].sum() / df.shape[0]
    categories_pct = categories_pct.sort_values(ascending=False)
    categories_list = list(categories_pct.index)

    # Word frequency
    word_freq = []
    for message in df['message'].values:
        word_freq.extend(tokenize(message))

    words = []
    counts = []
    word_dict = Counter(word_freq)
    word_list = list(word_dict.most_common(10))
    for item in word_list:
        words.append(item[0])
        counts.append(item[1])

    words_pct = list(np.array(counts) / df.shape[0] * 100)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            },
        },
        {
            'data': [
                Bar(
                    x=categories_list,
                    y=categories_pct
                )
            ],

            'layout': {
                'title': 'Messages by Category',
                'yaxis': {
                    'title': "Percentage",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -90,
                    'automargin':True
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=words_pct
                )
            ],

            'layout': {
                'title': 'Occurrence of top 10 words as a % of all words',
                'yaxis': {
                    'title': 'Occurrence',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 10 words',
                    'automargin': True
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()