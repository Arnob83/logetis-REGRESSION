import sqlite3
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import requests

# Paths to the model and scaler files
MODEL_URL = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/Logistic_Regression_model.pkl"
SCALER_URL = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/scaler.pkl"

MODEL_PATH = "Logistic_Regression_model.pkl"
SCALER_PATH = "scaler.pkl"

# Function to download files from a URL and save them locally
def download_file(url, local_filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file from {url}")

# Download the model and scaler if they do not exist
if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH)

if not os.path.exists(SCALER_PATH):
    download_file(SCALER_URL, SCALER_PATH)

# Load the trained model and feature names
with open(MODEL_PATH, 'rb') as model_file:
    loaded_model_dict = pickle.load(model_file)
    classifier = loaded_model_dict['model']
    trained_features = loaded_model_dict['feature_names']

# Load the scaler
with open(SCALER_PATH, 'rb') as scaler_file:
    scaler_dict = pickle.load(scaler_file)
    scaler = scaler_dict['scaler']

# Function to initialize the SQLite database
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

# Save prediction data to the database
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
def prediction(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Map user inputs to numeric values
    Education_1 = 0 if Education_1 == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data as a DataFrame with the correct column names
    input_data = pd.DataFrame(
        [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=trained_features  # Ensuring column names match the trained features
    )

    # Ensure the input data has the correct feature names, even if it's in a different order
    input_data = input_data[trained_features]  # This ensures the order of features is correct

    # Scale the input data using the scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict the result
    prediction = classifier.predict(input_data_scaled)
    probabilities = classifier.predict_proba(input_data_scaled)[0]  # Get probabilities

    return prediction[0], probabilities, input_data, input_data_scaled

# Main Streamlit app
def main():
    # Initialize database
    init_db()

    # App layout
    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f4f6f9;
            border: 2px solid #e6e8eb;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            background-color: #4caf50;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .header h1 {
            color: white;
        }
        </style>
        <div class="main-container">
        <div class="header">
        <h1>Loan Prediction ML App</h1>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
        # Run prediction
        result, probabilities, input_data, input_data_scaled = prediction(
            Credit_History,
            Education_1,
            ApplicantIncome,
            CoapplicantIncome,
            Loan_Amount_Term
        )

        # Save to database
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area,
                         Credit_History, Education_1, ApplicantIncome, CoapplicantIncome,
                         Loan_Amount_Term, "Approved" if result == 1 else "Rejected")

        # Display the prediction
        if result == 1:
            st.success(f"Your loan is Approved! (Probability: {probabilities[1]:.2f})", icon="✅")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0]:.2f})", icon="❌")

        # Show prediction values and scaled values
        st.subheader("Prediction Value")
        st.write(input_data)

        st.subheader("Input Data (Scaled)")
        st.write(pd.DataFrame(input_data_scaled, columns=trained_features))

        # Calculate feature contributions
        coefficients = classifier.coef_[0]
        feature_contributions = coefficients * input_data_scaled[0]

        # Create a DataFrame for visualization
        feature_df = pd.DataFrame({
            'Feature': trained_features,
            'Contribution': feature_contributions
        }).sort_values(by="Contribution", ascending=False)

        # Plot feature contributions
        st.subheader("Feature Contributions")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['green' if val >= 0 else 'red' for val in feature_df['Contribution']]
        ax.barh(feature_df['Feature'], feature_df['Contribution'], color=colors)
        ax.set_xlabel("Contribution to Prediction")
        ax.set_ylabel("Features")
        ax.set_title("Feature Contributions to Prediction")
        st.pyplot(fig)

        # Add explanations for the features
        st.subheader("Feature Contribution Explanations")
        for index, row in feature_df.iterrows():
            if row['Contribution'] >= 0:
                explanation = f"The feature '{row['Feature']}' positively influenced the loan approval."
            else:
                explanation = f"The feature '{row['Feature']}' negatively influenced the loan approval."
            st.write(f"- {explanation}")

    # Add a download button for the SQLite database
    if st.button("Download Database"):
        if os.path.exists("loan_data.db"):
            with open("loan_data.db", "rb") as db_file:
                st.download_button(
                    label="Download SQLite Database",
                    data=db_file,
                    file_name="loan_data.db",
                    mime="application/octet-stream"
                )
        else:
            st.error("Database file not found. Please try predicting a loan first to create the database.")

if __name__ == '__main__':
    main()
