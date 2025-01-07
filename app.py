import sqlite3
import pickle
import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt

# URLs to the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/logetis-REGRESSION/main/Logistic_Regression_model.pkl"

# Download the Logistic Regression model file and save it locally
response = requests.get(model_url)
with open("Logistic_Regression_model.pkl", "wb") as file:
    file.write(response.content)

# Load the trained model and feature names
with open("Logistic_Regression_model.pkl", "rb") as pickle_in:
    loaded_model_dict = pickle.load(pickle_in)
    classifier = loaded_model_dict['model']  # The trained Logistic Regression model
    trained_features = loaded_model_dict['feature_names']  # Extract the feature names

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
        credit_history REAL,
        education REAL,
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
def prediction(input_data):
    # Predict the result using the classifier
    pred = classifier.predict(input_data)
    pred_label = 'Approved' if pred[0] == 1 else 'Rejected'
    return pred_label

# Explain prediction using SHAP
def explain_prediction(input_data, final_result):
    explainer = shap.Explainer(classifier, input_data)
    shap_values = explainer(input_data)

    explanation_text = f"**Why your loan is {final_result}:**\n\n"
    for feature, shap_value in zip(input_data.columns, shap_values.values[0]):
        explanation_text += (
            f"- **{feature}**: {'Positive' if shap_value > 0 else 'Negative'} contribution with a SHAP value of {shap_value:.4f}\n"
        )

    if final_result == 'Rejected':
        explanation_text += "\nThe loan was rejected because the negative contributions outweighed the positive ones."
    else:
        explanation_text += "\nThe loan was approved because the positive contributions outweighed the negative ones."

    # Generate the SHAP bar plot
    plt.figure(figsize=(8, 5))
    colors = ['green' if value > 0 else 'red' for value in shap_values.values[0]]
    plt.barh(input_data.columns, shap_values.values[0], color=colors)
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.ylabel("Features")
    plt.title("Feature Contributions to Prediction")
    plt.tight_layout()

    return explanation_text, plt

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
    Credit_History = st.number_input("Credit History (0: Unclear Debts, 1: Clear Debts)", min_value=0.0, max_value=1.0)
    Education_1 = st.number_input("Education (0: Graduate, 1: Under_Graduate)", min_value=0, max_value=1)
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        # Prepare input data for prediction
        input_data = pd.DataFrame(
            [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
            columns=trained_features
        )

        # Predict and save the result
        result = prediction(input_data)
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        # Display the prediction
        if result == "Approved":
            st.success(f'Your loan is {result}', icon="✅")
        else:
            st.error(f'Your loan is {result}', icon="❌")

        # Explain the prediction
        st.header("Explanation of Prediction")
        explanation_text, bar_chart = explain_prediction(input_data, result)
        st.write(explanation_text)
        st.pyplot(bar_chart)

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
