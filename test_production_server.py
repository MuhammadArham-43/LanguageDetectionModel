import requests

headers = {
    'Content-Type': 'application/json',
}

res = requests.post('http://18.191.89.170:8000/predict', json={'text':"Esta é uma frase em inglês"}, headers=headers)

print(res.json())
