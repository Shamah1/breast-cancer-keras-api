# Breast Cancer Classification — Keras + FastAPI

Binary classification model to predict whether a tumor is **Benign** or **Malignant**, with a deployed REST API.

## Dataset
- Breast Cancer Wisconsin — 569 samples, 30 features
- Source: `sklearn.datasets.load_breast_cancer`
- Classes: 0 = Malignant (212), 1 = Benign (357)
- Split: 75% train / 25% test

## Project Structure
```
Keras-Classification.ipynb
deploy.py
model_weights.pkl
scaler_weights.pkl
```

## Model

Architecture: `Dense(30) → Dropout(0.2) → Dense(15) → Dropout(0.2) → Dense(1)`

```python
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
```

- Scaler: `MinMaxScaler`
- Regularization: Early Stopping (`patience=25`, monitor=`val_loss`) + Dropout
- Stopped at epoch 25

## Results

| | Precision | Recall | F1 |
|--|-----------|--------|----|
| Malignant (0) | 0.93 | 0.95 | 0.94 |
| Benign (1) | 0.97 | 0.95 | 0.96 |
| **Accuracy** | | | **0.95** |

**Confusion Matrix:**
```
[[52  3]
 [ 4 84]]
```

## API

Run with:
```bash
uvicorn deploy:app --reload
```

### Endpoints

**POST** `/predict` — send 30 features, saves prediction internally

**GET** `/result` — returns last saved prediction

**GET** `/evaluate` — returns model performance metrics on test set
