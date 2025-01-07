import sqlite3
import pickle
import streamlit as st
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# URLs to the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/Logistic_Regression_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/scaler.pkl"

# Download the Logistic Regression model file and save it locally
response = requests.get(model_url)
with open("Logistic_Regression_model.pkl", "wb") as file:
    file.write(response.content)

# Download the scaler file and save it locally
response = requests.get(scaler_url)
with open("scaler.pkl", "wb") as file:
    file.write(response.content)

# Load the trained model and feature names
with open("Logistic_Regression_model.pkl", "rb") as pickle_in:
    loaded_model_dict = pickle.load(pickle_in)
    classifier = loaded_model_dict['model']  # The trained Logistic Regression model
    trained_features = loaded_model_dict['feature_names']  # Extract the feature names

# Load the scaler used during training
with open("scaler.pkl", "rb") as scaler_file:
    scaler_dict = pickle.load(scaler_file)  # Load the dictionary
    scaler = scaler_dict['scaler']  # Extract the scaler object

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
@st.cache_data
def prediction(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Map user inputs to numeric values
    Education_1 = 0 if Education_1 == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data as a DataFrame
    input_data = pd.DataFrame(
        [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    )

    # Reorder and filter columns to match trained features
    input_data_filtered = input_data[trained_features]

    # Apply scaling using the loaded scaler
    input_data_scaled = scaler.transform(input_data_filtered)

    # Model prediction (0 = Rejected, 1 = Approved)
    prediction = classifier.predict(input_data_scaled)
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data_filtered

# LIME explanation function
def lime_explain(input_data_filtered, scaled_data, classifier):
    explainer = LimeTabularExplainer(
        training_data=scaled_data,
        feature_names=trained_features,
        class_names=["Rejected", "Approved"],
        mode="classification"
    )

    # Generate explanation for the first input instance
    exp = explainer.explain_instance(
        data_row=scaled_data[0],
        predict_fn=classifier.predict_proba
    )

    # Plot the explanation
    fig = exp.as_pyplot_figure()
    return fig

# Main Streamlit app
def main():
    # Initialize database
    init_db()

    st.title("Loan Prediction ML App")

    # User inputs
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", (0, 1, 2, 3, 4, 5))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semi-urban"))
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Education_1 = st.selectbox('Education', ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        result, input_data = prediction(
            Credit_History,
            Education_1,
            ApplicantIncome,
            CoapplicantIncome,
            Loan_Amount_Term
        )

        # Save data to database
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        # Display the prediction
        if result == "Approved":
            st.success(f'Your loan is {result}', icon="✅")
        else:
            st.error(f'Your loan is {result}', icon="❌")

        # Explain the prediction using LIME
        st.header("Explanation of Prediction")
        scaled_data = scaler.transform(input_data)  # Scale the input data
        lime_plot = lime_explain(input_data, scaled_data, classifier)
        st.pyplot(lime_plot)

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
