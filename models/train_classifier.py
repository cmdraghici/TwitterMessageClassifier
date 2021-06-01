import sys
from sqlalchemy import create_engine
import sqlalchemy
import pandas as pd
import re
import nltk
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM etl", engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    return X, Y, Y.columns


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = word_tokenize(text)
    words = [w for w in text if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    return stemmed


def build_model():
    pipeline = Pipeline([("vect", CountVectorizer(tokenizer=tokenize)),
                    ("tfidf", TfidfTransformer()),
                    ("clf", MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'tfidf__smooth_idf':[True, False],
        'clf__estimator__max_depth':[2, 7]
    }

    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_test = pd.DataFrame(Y_test)
    Y_pred = pd.DataFrame(Y_pred)

    for i in range(Y_test.shape[1]):
        print(classification_report(Y_test.iloc[i, :], Y_pred.iloc[i, :]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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