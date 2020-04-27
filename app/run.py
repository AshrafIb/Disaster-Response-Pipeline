import json
import plotly
import nltk
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)




nltk.download(['punkt','wordnet','stopwords'])
stop_english=stopwords.words('english')


def tokenize(text):
    '''
    Tokenizing Text-Input 
    
    Input: 
        Text
    Output: 
        Cleaned and tokenized text 
    
    '''
      
    word = ' '.join([i for i in text])
    # remove Punctuation
    word = re.sub(r'[^\w\s]','',word)
    # remove digits
    word = re.sub("\d+", " ", word)
    # lower words
    word = word.lower()
    # tokenize
    tokens = nltk.word_tokenize(word) 
    
    # removing stopwords 
    tokens_stop=[i for i in tokens if not i in stop_english]
    
    # lemmatizer 
    lemmatizer = WordNetLemmatizer()
    

    # lemmatize tokens 
    clean_tokens=[]
    
    for t in tokens_stop:
        clean=lemmatizer.lemmatize(t)
        clean_tokens.append(clean)  
        
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Text_clean', engine)


# load model
model = joblib.load("../models/classifier.pkl")


def freq_words(x, terms = 10): 
    '''
    Input: 
    x= Text Input 
    
    Terms= Number of Top-Words for Plot 
    
    Return: 
    Plot of Top-Words
    
    '''
    
    # Word Cleaning 
    word = ' '.join([text for text in x]) 
    # Remove punctuation
    word = re.sub(r'[^\w\s]','',word)
    # remove digits 
    word = re.sub("\d+", " ", word)
    # to lower 
    word = word.lower()
    # split 
    word = word.split() 
    
    # removing stopwords 
    word=[i for i in word if not i in stop_english]
    
    # Frequency of words 
    freqdist = nltk.FreqDist(word) 
    # To Dataframe
    words_df = pd.DataFrame({'word':list(freqdist.keys()), 'count':list(freqdist.values())})

    # Top50-Frame
    d = words_df.nlargest(columns="count", n = terms) 
    
    return d 


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = pd.DataFrame(df.groupby('genre').count()['message'])
    genre_counts = genre_counts.sort_values(by='message',ascending=False)
    genre_counts = genre_counts['message']
    genre_names = list(genre_counts.index)
    
    top10 = freq_words(df['message'])
    top10 =top10.sort_values(by='count',ascending=False)
    
    top10_counts=top10['count']
    top10_words=top10['word']
    
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
            }
        },
                
       {
        'data': [
                Bar(
                    x=top10_words,
                    y=top10_counts
                )
            ],

            'layout': {
                'title': 'Top10 Words in given Data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
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
    app.run(debug=True)


if __name__ == '__main__':
    main()