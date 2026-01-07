# predict_dropout.py
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_dropout():
    print("Please enter the following details (values in brackets are the valid ranges):")
    tuition_fees = float(input("Tuition fees up to date (0 or 1): "))
    curricular_units_1st_approved = int(input("Curricular units 1st sem (approved) (0-10): "))
    curricular_units_1st_grade = int(input("Curricular units 1st sem (grade) (0-20): "))
    curricular_units_2nd_approved = int(input("Curricular units 2nd sem (approved) (0-10): "))
    curricular_units_2nd_grade = int(input("Curricular units 2nd sem (grade) (0-20): "))

    input_data = [tuition_fees, curricular_units_1st_approved, curricular_units_1st_grade, curricular_units_2nd_approved, curricular_units_2nd_grade]
    print(f"Input data: {input_data}")

    # Preprocess the input data
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data)
    result = "Dropout" if prediction[0] == 1 else "Non-Dropout"
    print(f"The prediction for the provided data is: {result}")

# Example Usage of Real-time Prediction Function
if __name__ == "__main__":
    predict_dropout()
