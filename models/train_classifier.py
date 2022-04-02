# imports
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
import time
import warnings
warnings.filterwarnings('ignore')


def load_data_from_db(db_filename):
    """
    A function to load data from an SQLITE database into a Pandas dataframe
    :param db_filename: The name of the SQLITE database file
    :return: The following:
    *   X: (pandas.core.frame.DataFrame) containing features
    *   Y: (pandas.core.frame.DataFrame) containing target
    *   category_name: (list) containing labels for target
    """

    engine = create_engine('sqlite:///{}'.format(db_filename))
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    # The column Y['related'] contains three distinct values: 0, 1 and 2
    # Mapping the value 2 to 1
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)

    return X, Y, category_names


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


def build_model():
    """
    A function that builds a model using GridSearchCV
    :return: (sklearn.model_selection._search.GridSearchCV) trained model
    """

    # Build a model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])

    # Setup a hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # Create trained model
    model = GridSearchCV(estimator=pipeline,
                         param_grid=parameters,
                         verbose=3,
                         cv=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    A function that evaluates a given model
    :param model: (sklearn.model_selection._search.GridSearchCV) input model
    :param X_test: (pandas.core.series.Series) X_test values
    :param Y_test: (pandas.core.frame.DataFrame) Y_test values
    :param category_names: (pandas.core.indexes.base.Index) Names of categories
    :return: None
    """

    # Predict the model given X_test values
    y_pred = model.predict(X_test)

    # Print the classification report
    print(classification_report(Y_test.values,
                                y_pred,
                                target_names=category_names))

    # Print the accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filename):
    """
    A function that saves a given model to a file
    :param model: (sklearn.model_selection._search.GridSearchCV) input model
    :param model_filename: (str) the path and file name to save the model
    :return: None
    """

    # Save the model to a pickle file
    pickle.dump(model, open(model_filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        db_filename, model_filename = sys.argv[1:]
        print('Loading data...\n\t...DATABASE: {}'.format(db_filename))
        X, Y, category_names = load_data_from_db(db_filename)
        # Letting the test_size take its default value of 0.25 by not setting it
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        print('Building the model...')
        model = build_model()
        
        print('Training the model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating the model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving the model...\n    MODEL: {}'.format(model_filename))
        save_model(model, model_filename)

        print('Completed saving the trained model')

    else:
        print('Usage:')
        print('\tpython train_classifier.py <DB File Name> <Model File Name>')
        print('Example:')
        print('\tpython train_classifier.py DisasterResponse.db classifier.pkl')
        print('Please be mindful of the sequence of the parameters')


if __name__ == '__main__':
    main()
