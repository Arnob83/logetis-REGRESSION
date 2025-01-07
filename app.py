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
        result, input_data_scaled = prediction(
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
