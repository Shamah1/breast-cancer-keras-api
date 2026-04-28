from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Load scaler
with open('scaler_weights.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test = scaler.transform(X_test)

# Load model
with open('model_weights.pkl', 'rb') as f:
    weights = pickle.load(f)

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.build(input_shape=(None, 30))
model.set_weights(weights)

app = FastAPI()
result = {}

class TumorData(BaseModel):
    features: list

@app.post('/predict')
def predict(data: TumorData):
    X = np.array(data.features).reshape(1, -1)
    X = scaler.transform(X)
    prediction = model.predict(X)[0][0]
    result['prediction'] = 'Malignant' if prediction > 0.5 else 'Benign'
    result['probability'] = round(float(prediction), 4)
    return {'message': 'Prediction saved successfully'}

@app.get('/result')
def get_result():
    if not result:
        return {'message': 'No predictions yet'}
    return result

@app.get('/evaluate')
def evaluate():
    predictions = (model.predict(X_test) > 0.5).astype(int)
    report = classification_report(y_test, predictions, output_dict=True)
    return {
        'accuracy': round(report['accuracy'], 4),
        'malignant_precision': round(report['1']['precision'], 4),
        'malignant_recall': round(report['1']['recall'], 4),
        'benign_precision': round(report['0']['precision'], 4),
        'benign_recall': round(report['0']['recall'], 4)
    }