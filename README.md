# Loan Approval Prediction using Machine Learning

## Overview

This project predicts whether a loan application will be **Approved** or **Rejected** using a Machine Learning model trained on applicant and financial information.

The model is built using the **Random Forest Classifier** and deployed as an interactive **Streamlit Web Application** for real-time predictions.

---

## 🌐 Live Demo

**Streamlit Application:**
[Paste your Streamlit Cloud URL here
](https://loanapprovalprediction-9zd2zaqyttpnu2whxf9sma.streamlit.app/)
---

## Project Structure

```text
Loan_Approval_Prediction/
│
├── .devcontainer/
├── images/
├── README.md
├── app.py
├── loan_approval_dataset.csv
├── loan_approval_ml_model.ipynb
├── loan_approval_ml_model.pkl
└── requirements.txt
```

---

## Dataset Features

The model uses the following input features:

* Number of Dependents
* Education
* Self Employed
* Annual Income
* Loan Amount
* Loan Term
* CIBIL Score
* Residential Assets Value
* Commercial Assets Value
* Luxury Assets Value
* Bank Asset Value

---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Train-Test Split
6. Model Training using Random Forest Classifier
7. Hyperparameter Tuning using RandomizedSearchCV
8. Model Evaluation
9. Feature Importance Analysis
10. Model Serialization using Joblib
11. Streamlit Web Application Deployment

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn
* Streamlit

---

## Model Performance

| Metric                 |      Score |
| ---------------------- | ---------: |
| Test Accuracy          | **97.19%** |
| Cross Validation Score | **97.98%** |

### Evaluation Metrics Used

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Feature Importance

---

## Feature Importance

The Random Forest model identified the following features as the most influential in predicting loan approval:

| Rank | Feature     |
| ---: | ----------- |
|    1 | CIBIL Score |
|    2 | Loan Term   |
|    3 | Loan Amount |

**Observation:**
CIBIL Score is the most influential feature used by the model for predicting loan approval.

---

## Streamlit Application

The Streamlit application allows users to:

* Enter applicant details through an interactive interface
* Predict whether a loan will be approved or rejected
* Display prediction probabilities
* Provide real-time predictions using the trained Machine Learning model

---

## Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/AryanChaudhari8457/Loan_Approval_Prediction.git
```

### Navigate to the project folder

```bash
cd Loan_Approval_Prediction
```

### Install the required libraries

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## Project Screenshots

### Home Page

<img width="1900" height="862" alt="Screenshot 2026-07-13 181022" src="https://github.com/user-attachments/assets/dcb50d11-83a9-401d-b1b3-c3cc79a6f667" />


### Prediction Result

<img width="1912" height="840" alt="Screenshot 2026-07-13 181115" src="https://github.com/user-attachments/assets/0d55e5aa-947b-4ff9-be25-5e62352cc24e" />


---

## Visualizations

This project includes:

* Feature Importance
* Confusion Matrix
* Correlation Heatmap
* Loan Status Distribution

---

## Future Improvements

* Compare multiple Machine Learning algorithms
* Add Explainable AI (SHAP/LIME)
* Deploy using Docker
* Connect with a database
* Add user authentication
* Support batch predictions

---

## Author

**Aryan Chaudhari**

---
