from fastapi import FastAPI
from pydantic import BaseModel
from model.prediction import predict_language

import uvicorn 
 
app = FastAPI()

class InputText(BaseModel):
    text: str

class OutputText(BaseModel):
    language: str
    
@app.get('/')
def home():
    return {'health check': 'ok'}

@app.post('/predict', response_model=OutputText)
def predict(payload:InputText):
    language = predict_language(payload.text)
    return {'language': language}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)