import sqlite3
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import requests

# URLs to the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/Logistic_Regression_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/scaler.pkl"

# Download and save model and scaler
for url, file_name in [(model_url, "Logistic_Regression_model.pkl"), (scaler_url, "scaler.pkl")]:
    response = requests.get(url)
    with open(file_name, "wb") as file:
        file.write(response.content)

# Load the model and scaler
with open("Logistic_Regression_model.pkl", "rb") as file:
    loaded_model_dict = pickle.load(file)
    classifier = loaded_model_dict["model"]
    trained_features = loaded_model_dict["feature_names"]

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)["scaler"]

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender TEXT,
        married TEXT,
        dependents INTEGER,
        self_employed TEXT,
        loan_amount REAL,
        property_area TEXT,
        credit_history TEXT,
        education TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount_term REAL,
        result TEXT
    )
    """)
    conn.commit()
    conn.close()

# Save data to the database
def save_to_database(gender, married, dependents, self_employed, loan_amount, property_area, 
                     credit_history, education, applicant_income, coapplicant_income, 
                     loan_amount_term, result):
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO loan_predictions (
        gender, married, dependents, self_employed, loan_amount, property_area, 
        credit_history, education, applicant_income, coapplicant_income, loan_amount_term, result
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (gender, married, dependents, self_employed, loan_amount, property_area, 
          credit_history, education, applicant_income, coapplicant_income, 
          loan_amount_term, result))
    conn.commit()
    conn.close()

# Prediction function
@st.cache_data
def prediction(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Map user inputs to numeric values
    Credit_History = 1 if Credit_History == "Clear Debts" else 0
    Education_1 = 0 if Education_1 == "Graduate" else 1

    # Create input data as a DataFrame
    input_data = pd.DataFrame(
        [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    )

    # Reorder and filter columns to match trained features
    input_data = input_data[trained_features]  # Ensure proper column alignment

    # Apply scaling using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Model prediction (0 = Rejected, 1 = Approved)
    prediction = classifier.predict(input_data_scaled)
    pred_label = "Approved" if prediction[0] == 1 else "Rejected"
    return pred_label, input_data

# Explain prediction using LIME
def explain_prediction(input_data, result):
    explainer = LimeTabularExplainer(
        training_data=scaler.transform(pd.DataFrame(columns=trained_features, index=range(1))),  # Placeholder
        feature_names=trained_features,
        class_names=["Rejected", "Approved"],
        mode="classification"
    )
    explanation = explainer.explain_instance(
        data_row=input_data.iloc[0].to_numpy(),
        predict_fn=classifier.predict_proba
    )
    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)
    explanation_text = f"**Why your loan is {result}:**\n\n"
    for feature, weight in explanation.as_list():
        contribution = "Positive" if weight > 0 else "Negative"
        explanation_text += f"- **{feature}**: {contribution} contribution with a weight of {weight:.4f}\n"
    st.write(explanation_text)

# Main Streamlit app
def main():
    init_db()
    st.title("Loan Prediction ML App")
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4, 5])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semi-urban"])
    Credit_History = st.selectbox("Credit History", ["Unclear Debts", "Clear Debts"])
    Education_1 = st.selectbox("Education", ["Under_Graduate", "Graduate"])
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        result, input_data = prediction(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term)
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, result)
        st.success(f"Your loan is {result}" if result == "Approved" else f"Your loan is {result}", icon="✅")
        explain_prediction(input_data, result)

if __name__ == "__main__":
    main()
