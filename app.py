import streamlit as st
import joblib

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------------------
# Load Model
# -----------------------------------------
model = joblib.load("loan_approval_ml_model.pkl")

# -----------------------------------------
# Title
# -----------------------------------------
st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below and click **Predict Loan**.")

st.divider()

# -----------------------------------------
# User Inputs
# -----------------------------------------

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

    income_annum = st.number_input(
        "Annual Income",
        min_value=0,
        value=500000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=100000
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        value=10
    )

with col2:

    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=750
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value",
        min_value=0,
        value=1000000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value",
        min_value=0,
        value=0
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value",
        min_value=0,
        value=0
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value",
        min_value=0,
        value=500000
    )

# -----------------------------------------
# Convert Inputs
# -----------------------------------------

education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -----------------------------------------
# Prediction Button
# -----------------------------------------

if st.button("🔍 Predict Loan", use_container_width=True):

    new_data = [[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]]

    prediction = model.predict(new_data)

    probability = model.predict_proba(new_data)

    st.divider()

    if prediction[0] == 1:
        st.success("✅ Loan Approved")

    else:
        st.error("❌ Loan Rejected")

    st.subheader("Prediction Probability")

    st.write(f"**Approved :** {probability[0][1]*100:.2f}%")
    st.write(f"**Rejected :** {probability[0][0]*100:.2f}%")

    st.progress(float(probability[0][1]))