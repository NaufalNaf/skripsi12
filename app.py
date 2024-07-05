import joblib
import pandas as pd
from sklearn import preprocessing

model = joblib.load('klasifikasi.sav')
scaler = joblib.load('scaler.sav')

gender_value = {'Male': 0, 'Female': 1}
index_labels = {
    0: 'Extremely Weak',
    1: 'Weak',
    2: 'Normal',
    3: 'Overweight',
    4: 'Obese',
    5: 'Extremely Obese'
}

def predict(input_data):
    # Convert input data to a Pandas DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Replace categorical values with numerical values
    input_df['Gender'] = input_df['Gender'].replace(gender_value)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Predict the BMI category
    prediction = model.predict(input_scaled)[0]

    # Return the prediction and the corresponding BMI category label
    return prediction, index_labels[prediction]

def main():
    # Get input data from the user
    gender = input("Enter your gender (Male/Female): ")
    height = float(input("Enter your height (cm): "))
    weight = float(input("Enter your weight (kg): "))

    # Create input data dictionary
    input_data = {
        'Gender': gender,
        'Height': height,
        'Weight': weight
    }

    # Predict the BMI category and label
    prediction, label = predict(input_data)

    # Print the prediction and label
    print(f"Predicted BMI Category: {label}")

if __name__ == "__main__":
    main()
