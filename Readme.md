# Breast Cancer Classification Task

Binary classification model to predict whether a tumor is **Benign** or **Malignant**.

## Dataset
Breast Cancer Wisconsin — 569 samples, 30 features, 2 classes (212 Malignant, 357 Benign)

## Project Structure
```
keras-classification-task.ipynb
deploy.py
model_weights.pkl
scaler_weights.pkl
```

## Model
- Architecture: Dense(30) → Dropout(0.2) → Dense(15) → Dropout(0.2) → Dense(1)
- Optimizer: Adam | Loss: Binary Crossentropy
- Techniques: Early Stopping + Dropout
- Accuracy: ~95%

## Run API
```
uvicorn deploy:app --reload
```

## Endpoints

**POST** `/predict` — send 30 features, saves prediction internally

**GET** `/result` — returns last saved prediction

**GET** `/evaluate` — returns model performance metrics
