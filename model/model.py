import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from preprocess import preprocess_text


if __name__ == "__main__":
    DATA_FILE_PATH = '../data/Language Detection.csv'
    
    data = pd.read_csv(DATA_FILE_PATH)
    texts = data['Text']
    labels = data['Language']
    
    labelEncoder = LabelEncoder()
    labels = labelEncoder.fit_transform(labels)
    
    data = list(map(lambda x: preprocess_text(x), texts))
        
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
    
    cv = CountVectorizer()
    model = MultinomialNB()
    
    print('Training Model...')
    
    
    pipe = Pipeline([('vectorizer', cv), ('naiveBeyes', model)])
    pipe.fit(train_data, train_labels)
    prediction = pipe.predict(test_data)
    print(accuracy_score(prediction, test_labels) * 100)
    
    with open('trained_language_model.pkl', 'wb') as f:
        pickle.dump(pipe, f)