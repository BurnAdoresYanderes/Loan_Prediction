import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set the page configuration
st.set_page_config(page_title='Loan Prediction App', layout='centered')

# --- Model and Data Loading ---
@st.cache_data
def load_model_and_data():
    """Load the trained model and training columns."""
    model = joblib.load('tuned_random_forest_model.pkl')
    # Load the training data to get the column order
    training_data = pd.read_csv('loan_data.csv')
    training_data = training_data[training_data['person_age'] < 100]
    training_data['previous_loan_defaults_on_file'] = training_data['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    training_data = pd.get_dummies(training_data, columns=['person_gender', 'person_education', 'person_home_ownership', 'loan_intent'], drop_first=True)
    training_columns = training_data.drop('loan_status', axis=1).columns.tolist()
    return model, training_columns

model, training_columns = load_model_and_data()


# --- App Header ---
st.title('Loan Approval Predictor')
st.write("Enter the applicant's details to predict loan approval status.")


# --- User Input Form ---
with st.form("loan_form"):
    person_age = st.number_input('Age', min_value=18, max_value=100, value=25)
    person_income = st.number_input('Annual Income', min_value=0, value=50000)
    person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    person_emp_exp = st.number_input('Employment Experience (years)', min_value=0, max_value=50, value=5)
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_amnt = st.number_input('Loan Amount', min_value=0, value=10000)
    loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, max_value=30.0, value=10.0, format="%.2f")
    previous_loan_defaults_on_file = st.selectbox('Previous Defaults', ['No', 'Yes'])
    
    # Advanced options in an expander
    with st.expander("Optional Advanced Details"):
        loan_percent_income = st.number_input('Loan as Percent of Income', min_value=0.0, max_value=1.0, value=0.2, format="%.2f")
        cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, max_value=40, value=4)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
        person_gender = st.selectbox('Gender', ['Male', 'Female'])
        person_education = st.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'Associate', 'Professional', 'Doctorate'])

    submitted = st.form_submit_button('Predict')


# --- Prediction Logic and Display ---
if submitted:
    # --- Data Preprocessing for Prediction ---
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': 1 if previous_loan_defaults_on_file == 'Yes' else 0,
        'person_gender_male': 1 if person_gender == 'Male' else 0,
        'person_education_Bachelor': 1 if person_education == 'Bachelor' else 0,
        'person_education_Doctorate': 1 if person_education == 'Doctorate' else 0,
        'person_education_High School': 1 if person_education == 'High School' else 0,
        'person_education_Master': 1 if person_education == 'Master' else 0,
        'person_education_Professional': 1 if person_education == 'Professional' else 0,
        'person_home_ownership_OWN': 1 if person_home_ownership == 'OWN' else 0,
        'person_home_ownership_RENT': 1 if person_home_ownership == 'RENT' else 0,
        'person_home_ownership_OTHER': 1 if person_home_ownership == 'OTHER' else 0,
        'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0
    }

    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Ensure all training columns are present and in the correct order
    final_df = pd.DataFrame(columns=training_columns)
    final_df = pd.concat([final_df, input_df], ignore_index=True).fillna(0)
    final_df = final_df[training_columns] # Enforce column order

    # --- Prediction ---
    prediction = model.predict(final_df)[0]
    prediction_proba = model.predict_proba(final_df)[0]

    # --- Display Results ---
    st.write('---')
    st.header('Prediction Result')

    # This logic is now correct based on the data provided
    if prediction == 1: # 1 means APPROVED
        st.success('**Loan Approved!** 👍')
        st.write(f"Confidence Score: {prediction_proba[1]*100:.2f}%")
        st.balloons()
    else: # 0 means REJECTED
        st.error('**Loan Rejected!** 👎')
        st.write(f"Confidence Score (for rejection): {prediction_proba[0]*100:.2f}%")

    with st.expander("View Probabilities"):
        st.write({
            'Probability of Rejection (Status 0)': prediction_proba[0],
            'Probability of Approval (Status 1)': prediction_proba[1]
        })