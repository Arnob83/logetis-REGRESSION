import sqlite3
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import requests

# Paths to the model and scaler files
MODEL_PATH = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/Logistic_Regression_model.pkl"
SCALER_PATH = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/scaler.pkl"

# Load the trained model and feature names
def load_model():
    try:
        model_response = requests.get(MODEL_PATH)
        model_response.raise_for_status()
        loaded_model_dict = pickle.loads(model_response.content)
        classifier = loaded_model_dict["model"]
        trained_features = loaded_model_dict["feature_names"]
        return classifier, trained_features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the scaler
def load_scaler():
    try:
        scaler_response = requests.get(SCALER_PATH)
        scaler_response.raise_for_status()
        scaler_dict = pickle.loads(scaler_response.content)
        scaler = scaler_dict["scaler"]
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()

classifier, trained_features = load_model()
scaler = load_scaler()

# Function to initialize the SQLite database
def init_db():
    conn = sqlite3.connect(":memory:")  # In-memory database for Streamlit Cloud
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
    return conn

conn = init_db()

# Save prediction data to the database
def save_to_database(conn, gender, married, dependents, self_employed, loan_amount, property_area,
                     credit_history, education, applicant_income, coapplicant_income,
                     loan_amount_term, result):
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

# Prediction function
def prediction(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Map user inputs to numeric values
    Education_1 = 0 if Education_1 == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data as a DataFrame
    input_data = pd.DataFrame(
        [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    )

    # Reorder columns to match trained features
    input_data_filtered = input_data[trained_features]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_filtered)

    # Predict the result
    prediction = classifier.predict(input_data_scaled)
    probabilities = classifier.predict_proba(input_data_scaled)[0]  # Get probabilities

    return prediction[0], probabilities, input_data_filtered, input_data_scaled

# Main Streamlit app
def main():
    st.title("Loan Prediction ML App")

    # User inputs
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", (0, 1, 2, 3, 4, 5))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semi-urban"))
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Education_1 = st.selectbox("Education", ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        # Validate inputs
        if Loan_Amount <= 0 or ApplicantIncome <= 0 or Loan_Amount_Term <= 0:
            st.error("Please provide valid inputs for Loan Amount, Income, and Loan Term.")
            return

        # Run prediction
        result, probabilities, input_data, input_data_scaled = prediction(
            Credit_History,
            Education_1,
            ApplicantIncome,
            CoapplicantIncome,
            Loan_Amount_Term
        )

        # Save to database
        save_to_database(conn, Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area,
                         Credit_History, Education_1, ApplicantIncome, CoapplicantIncome,
                         Loan_Amount_Term, "Approved" if result == 1 else "Rejected")

        # Display the prediction
        if result == 1:
            st.success(f"Your loan is Approved! (Probability: {probabilities[1]:.2f})")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0]:.2f})")

        # Show prediction values and scaled values
        st.subheader("Prediction Value")
        st.write(input_data)

        st.subheader("Input Data (Scaled)")
        st.write(pd.DataFrame(input_data_scaled, columns=trained_features))

        # Plot feature contributions
        coefficients = classifier.coef_[0]
        feature_contributions = coefficients * input_data_scaled[0]

        feature_df = pd.DataFrame({
            'Feature': trained_features,
            'Contribution': feature_contributions
        }).sort_values(by="Contribution", ascending=False)

        st.subheader("Feature Contributions")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['green' if val >= 0 else 'red' for val in feature_df['Contribution']]
        ax.barh(feature_df['Feature'], feature_df['Contribution'], color=colors)
        ax.set_xlabel("Contribution to Prediction")
        ax.set_ylabel("Features")
        ax.set_title("Feature Contributions to Prediction")
        st.pyplot(fig)

if __name__ == '__main__':
    main()
