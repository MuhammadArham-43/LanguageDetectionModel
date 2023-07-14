import re

def preprocess_text(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub("[[]]", " ", text)
    text = text.lower()
    return text
