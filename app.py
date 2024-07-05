import joblib
import pandas as pd
from sklearn import preprocessing

model = joblib.load('models/klasifikasi.sav')
scaler = joblib.load('models/scaler.sav')

index_labels = {
    0: 'Extremely Weak',
    1: 'Weak',
    2: 'Normal',
    3: 'Overweight',
    4: 'Obese',
    5: 'Extremely Obese'
}

gender_mapping = {
    0: 'Male',
    1: 'Female'
}

def predict(gender, height, weight):
    data = [[gender, height, weight]]
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    return index_labels[prediction]
