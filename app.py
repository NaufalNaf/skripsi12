import streamlit as st
import joblib
import numpy as np

# Load the saved models
scaler = joblib.load('scaler.sav')
model = joblib.load('klasifikasi.sav')

# Define the application layout and functionality
def main():
    st.title('BMI Classification App')

    # Get user input
    gender = st.selectbox('Gender', ['Male', 'Female'])
    height = st.number_input('Height (cm)', min_value=100, max_value=250)
    weight = st.number_input('Weight (kg)', min_value=10, max_value=200)

    # Preprocess the user input
    data = np.array([[gender, height, weight]])
    data_scaled = scaler.transform(data)

    # Make a prediction
    prediction = model.predict(data_scaled)[0]

    # Display the prediction
    index_labels = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obese',
        5: 'Extremely Obese'
    }
    st.write('Predicted BMI Category:', index_labels[prediction])

if __name__ == '__main__':
    main()
