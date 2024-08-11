import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Load the encoders and scaler
with open('onehot_encoder_gender.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('level_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI components
st.title("Churn Prediction")

geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
credit_score = st.number_input("Credit Score")
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Create a DataFrame from user input
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_df = pd.DataFrame([input_data])

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine with the original DataFrame
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Adjust the input shape if necessary
input_scaled_corrected = np.hstack([input_scaled, np.zeros((1, 2))])

# Make predictions
predictions = model.predict(input_scaled_corrected)
prediction_prob = predictions[0][0]

# Display the result
if prediction_prob > 0.5:
    st.write("The person is likely to churn.")
else:
    st.write("The person is unlikely to churn.")
