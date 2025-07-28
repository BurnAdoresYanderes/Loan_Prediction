import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODEL AND DATA LOADING ---
# Use caching to load the model and data only once
@st.cache_resource
def load_model_and_data():
    """Load the trained model and the original dataset to get column information."""
    try:
        model = joblib.load('tuned_random_forest_model.pkl')
        # Load the original data to get the exact column order for one-hot encoding
        df = pd.read_csv('loan_data.csv')
        # Basic preprocessing similar to the notebook
        df = df[df['person_age'] < 100]
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
        # One-hot encode to get the correct columns
        df_encoded = pd.get_dummies(df, columns=['person_gender', 'person_education', 'person_home_ownership', 'loan_intent'], drop_first=True)
        # Get the columns X was trained on (all except the target)
        training_columns = df_encoded.drop('loan_status', axis=1).columns
        return model, training_columns
    except FileNotFoundError:
        return None, None

model, training_columns = load_model_and_data()

# --- STYLES ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- WEB APPLICATION LAYOUT ---
if model is None or training_columns is None:
    st.error("Model file (`tuned_random_forest_model.pkl`) or data file (`loan_data.csv`) not found. Please ensure they are in the same directory.")
else:
    # --- HEADER ---
    st.title('🏦 Loan Approval Prediction App')
    st.write(
        "This application uses a Random Forest model to predict the likelihood of a loan being approved. "
        "Please enter the applicant's details in the sidebar to get a prediction."
    )

    # --- SIDEBAR FOR USER INPUT ---
    st.sidebar.header('Applicant Information')

    # Numerical Inputs
    age = st.sidebar.number_input('Age', min_value=18, max_value=99, value=25)
    income = st.sidebar.number_input('Annual Income ($)', min_value=4000, max_value=1000000, value=65000)
    emp_exp = st.sidebar.number_input('Employment Experience (Years)', min_value=0, max_value=50, value=5)
    loan_amnt = st.sidebar.number_input('Loan Amount ($)', min_value=500, max_value=50000, value=10000)
    int_rate = st.sidebar.slider('Loan Interest Rate (%)', min_value=5.0, max_value=25.0, value=11.0, step=0.1)
    credit_score = st.sidebar.slider('Credit Score', min_value=300, max_value=850, value=650)
    
    # Calculated field - handled in prediction logic
    loan_percent_income = (loan_amnt / income) if income > 0 else 0
    st.sidebar.markdown(f"**Loan as % of Income:** `{loan_percent_income:.2%}`")


    cb_hist_length = st.sidebar.number_input('Credit History Length (Years)', min_value=1, max_value=40, value=6)

    # Categorical Inputs
    home_ownership = st.sidebar.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    loan_intent = st.sidebar.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    education = st.sidebar.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'Associate', 'Other'])
    gender = st.sidebar.selectbox('Gender', ['male', 'female'])
    previous_defaults = st.sidebar.selectbox('Previous Defaults on File?', ['No', 'Yes'])


    # --- PREDICTION LOGIC ---
    if st.button('Predict Loan Status'):
        # Create a dictionary of the inputs
        input_data = {
            'person_age': age,
            'person_income': income,
            'person_emp_exp': emp_exp,
            'loan_amnt': loan_amnt,
            'loan_int_rate': int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_defaults,
            'person_gender': gender,
            'person_education': education,
            'person_home_ownership': home_ownership,
            'loan_intent': loan_intent,
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the data to match the model's training format
        # Map 'Yes'/'No' to 1/0
        input_df['previous_loan_defaults_on_file'] = input_df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
        
        # One-hot encode the categorical features
        input_df_encoded = pd.get_dummies(input_df)
        
        # Align the columns with the training data columns
        # This ensures all required columns are present and in the correct order
        input_df_aligned = input_df_encoded.reindex(columns=training_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df_aligned)
        probability = model.predict_proba(input_df_aligned)

        # --- DISPLAY RESULTS (Corrected Logic) ---
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            # Based on our analysis, loan_status=1 means Approved
            st.success('**Loan Status: Approved (Low risk)**', icon="✔️")
            prob_value = probability[0][1] # Probability of class 1 (Approved)
            st.markdown(f"<p class='big-font'>The model predicts a {prob_value:.2%} probability of this loan being approved.</p>", unsafe_allow_html=True)
            st.balloons()
        else:
            # Based on our analysis, loan_status=0 means Rejected
            st.error('**Loan Status: Rejected (High risk)**', icon="✖️")
            prob_value = probability[0][0] # Probability of class 0 (Rejected)
            st.markdown(f"<p class='big-font'>The model predicts a {prob_value:.2%} probability of this loan being rejected.</p>", unsafe_allow_html=True)

        # Display probability breakdown
        st.write("---")
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame({
            'Status': ['Rejected (Status 0)', 'Approved (Status 1)'],
            'Probability': [f"{probability[0][0]:.2%}", f"{probability[0][1]:.2%}"]
        })
        st.table(prob_df)