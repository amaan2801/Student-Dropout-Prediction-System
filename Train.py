# train_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Data
df = pd.read_csv("Dataset.csv", sep=";")

# Basic Data Info
print(df.info())
print(df.describe().T)
print(df['Target'].value_counts())

# Encode target variable
df['Target'] = LabelEncoder().fit_transform(df['Target'])
print(df['Target'].value_counts())

# Drop irrelevant target values
df = df[df['Target'] != 1]
df['Dropout'] = df['Target'].apply(lambda x: 1 if x == 0 else 0)

# Standard Scaling the Data
features = ["Tuition fees up to date", "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)", "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"]
x = df[features].values
scaler = StandardScaler().fit(x)
x = scaler.transform(x)
y = df['Dropout'].values

# Train & Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Train Random Forest Model
model_rf = RandomForestClassifier(n_estimators=500, criterion='entropy')
model_rf.fit(x_train, y_train)

# Save the trained model and scaler
joblib.dump(model_rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")
