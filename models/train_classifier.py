import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import nltk
import re
import time
import pickle

from sqlalchemy import create_engine 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV , ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# nltk additions 
nltk.download(['punkt','wordnet','stopwords'])

# stop words 
stop_english=stopwords.words('english')



def load_data(database_filepath,table_name='Text_clean'):
    '''
    Loading Data from sql-Database
    
    Input: 
        database_filepath: path to Database
    Outputs: 
        X: Features of ML model 
        Y: Target values for ML model
        label_names: Categories of Y 
    
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name, engine)
    X=df['message']
    Y=df.drop(['id','message','original','genre'],axis=1)
    # Changing wrongly labeld data
    Y['related']=np.where(Y['related']==2,1,Y['related'])
    Y['related']
    category_names=Y.columns
    return X,Y,category_names
    
  

def tokenize(text):
    '''
    Cleaning Text and building tokens
    
    Input:
        Text column
    
    Output:
        tokenized and cleaned text
       
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
    tokens_clean=[]
    
    for t in tokens_stop:
        clean=lemmatizer.lemmatize(t)
        tokens_clean.append(clean)  
        
    return tokens_clean
    

def build_model():
    
    '''
    Build a ML-pipeline with a CountVectorizer, TfidfTransformer
    and a MultioutputClassifier (RandomForestClassifier)
    
    The included params are the optimized parameters. 
    
    Input: 
        None
        
    Output:
        Machine-Learning Pipeline
    
    
    ''' 
    
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize,lowercase=False)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))    
    ])
    
    param_grid = { 
            "clf__estimator__min_samples_split": [2],
            "clf__estimator__criterion": ["entropy"],
            'clf__estimator__max_depth': [50],
            "clf__estimator__max_features" : ["auto"]
            }
    
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1,
                        cv=3, verbose=10)
    return grid 
    
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluating Model and printing means of recall, f1,accuracy and precision 
    
    Input: 
        model: Ml-Pipeline
        X_test: Test-split for Ml-predict
        Y_test: Testsplit for Ml-predict 
        category_names: label_names
        
    Output: 
        Printing Model-Evaluation
    
    
    '''
    
    # Predicting results 
    Y_pred = model.predict(X_test)

   
    # Lists for metrics
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    # iterate over scores
    
    for i in range(len(category_names)):
        
            accuracy = accuracy_score(Y_test[category_names[i]].values, Y_pred[:,i])
            precision = precision_score(Y_test[category_names[i]].values, Y_pred[:,i],average='weighted')
            recall = recall_score(Y_test[category_names[i]].values, Y_pred[:,i],average='weighted')
            f1 = f1_score(Y_test[category_names[i]].values, Y_pred[:,i],average='weighted') 
            
     # append to lists 
    
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            
    # create dictionary 
    
    
    scores = {'Label Names':category_names,'Accuracy':accuracy_list, 'Precision':precision_list, 
              'Recall':recall_list,'F1':f1_list}  
    
    # Create Dataframe
    df=pd.DataFrame.from_dict(scores)
    
    
    # Print 
    for i in df.columns:
        if i!='Label Names':
            print ('The overall {} score is {:.2%}'.format(i,round(df[i].mean(),6)))
    
    print(df)
    
    
    

def save_model(model, model_filepath):
    '''
    Saving Model 
    
    model: Estimator 
    model_filepath: Path to pickle 
    
    
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()