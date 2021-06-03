import re
import pandas as pd
import joblib
import plotly
import json

from flask import Flask
from flask import render_template, request
from sqlalchemy import create_engine
from plotly.graph_objs import Bar

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/disaster_database.db')
df = pd.read_sql_table('etl', engine)

# load model
model = joblib.load("../models/model.sav")


@app.route('/')
@app.route('/index')
def index():

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
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
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template('go.html', query=query, classification_result=classification_results)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()