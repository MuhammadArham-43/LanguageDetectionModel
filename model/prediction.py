import pickle
from pathlib import Path
import os

from .preprocess import preprocess_text
from .classes import classes

BASE_DIR = Path(__file__).resolve(strict=True).parent

BASE_MODEL_PATH = os.path.join(BASE_DIR, 'trained_language_model.pkl')

with open(BASE_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

def predict_language(text):
    text = preprocess_text(text)
    pred = model.predict([text])
    return classes[pred[0]]

if __name__ == "__main__":
    
    
    print(predict_language('This is an English Sentence'))