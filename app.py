import streamlit as st
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = joblib.load("loan_approval_ml_model.pkl")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("🏦 Loan Approval Predictor")

st.sidebar.info("""
This application predicts whether a loan application
will be **Approved** or **Rejected**
using a **Random Forest Machine Learning Model**.
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Model")
st.sidebar.success("Random Forest Classifier")

st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.write("Aryan Chaudhari")

st.sidebar.markdown("---")
st.sidebar.write("### Instructions")
st.sidebar.write("""
- Enter all applicant details.
- Click **Predict Loan Status**.
- View approval probability instantly.
""")

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🏦 Loan Approval Prediction System")

st.markdown("""
Predict whether a loan application is likely to be **Approved**
or **Rejected** using a Machine Learning model trained on historical loan data.
""")

st.divider()

# =================================================
# Applicant Information
# =================================================

with st.expander("👤 Applicant Information", expanded=True):

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

    with col2:

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

# =================================================
# Financial Information
# =================================================

with st.expander("💰 Financial Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        income_annum = st.number_input(
            "Annual Income (₹)",
            min_value=0,
            value=500000,
            step=50000
        )

        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0,
            value=100000,
            step=50000
        )

    with col2:

        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1,
            value=10
        )

        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=750
        )

# =================================================
# Asset Information
# =================================================

with st.expander("🏠 Asset Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        residential_assets_value = st.number_input(
            "Residential Assets (₹)",
            min_value=0,
            value=1000000,
            step=100000
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets (₹)",
            min_value=0,
            value=0,
            step=100000
        )

    with col2:

        luxury_assets_value = st.number_input(
            "Luxury Assets (₹)",
            min_value=0,
            value=0,
            step=100000
        )

        bank_asset_value = st.number_input(
            "Bank Assets (₹)",
            min_value=0,
            value=500000,
            step=50000
        )

# =================================================
# Convert Inputs
# =================================================

education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# =================================================
# Prediction Button
# =================================================

if st.button("🚀 Predict Loan Status", use_container_width=True):

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

    with st.spinner("Analyzing Loan Application..."):

        prediction = model.predict(new_data)
        probability = model.predict_proba(new_data)

    st.divider()

    # -------------------------------------------------
    # Result
    # -------------------------------------------------

    if prediction[0] == 1:

        st.balloons()

        st.success("🎉 Loan Approved")

        st.markdown("""
### Congratulations!

The applicant has a **high probability**
of getting the loan approved.
""")

    else:

        st.error("❌ Loan Rejected")

        st.markdown("""
### Suggestions

- Improve your CIBIL Score
- Reduce Loan Amount
- Increase Annual Income
- Improve Asset Value
""")

    # -------------------------------------------------
    # Probability
    # -------------------------------------------------

    st.subheader("📊 Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Approval Probability",
            f"{probability[0][1]*100:.2f}%"
        )

    with col2:

        st.metric(
            "Rejection Probability",
            f"{probability[0][0]*100:.2f}%"
        )

    st.write("### Model Confidence")

    st.progress(float(max(probability[0])))

    # -------------------------------------------------
    # Application Summary
    # -------------------------------------------------

    st.subheader("📋 Application Summary")

    st.table({
        "Feature": [
            "Dependents",
            "Education",
            "Self Employed",
            "Annual Income",
            "Loan Amount",
            "Loan Term",
            "CIBIL Score",
            "Residential Assets",
            "Commercial Assets",
            "Luxury Assets",
            "Bank Assets"
        ],
        "Value": [
            no_of_dependents,
            "Graduate" if education == 1 else "Not Graduate",
            "Yes" if self_employed == 1 else "No",
            f"₹{income_annum:,}",
            f"₹{loan_amount:,}",
            f"{loan_term} Years",
            cibil_score,
            f"₹{residential_assets_value:,}",
            f"₹{commercial_assets_value:,}",
            f"₹{luxury_assets_value:,}",
            f"₹{bank_asset_value:,}"
        ]
    })

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("---")

st.caption(
    "🏦 Loan Approval Prediction | Built with Streamlit & Scikit-learn | Developed by Aryan Chaudhari"
)
