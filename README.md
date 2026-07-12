# Loan Approval Prediction using Machine Learning

## Project Overview

This project predicts whether a loan application will be **Approved** or **Rejected** using Machine Learning.

The model was built using a **Random Forest Classifier** and improved through **Hyperparameter Tuning (RandomizedSearchCV)**. The final trained model is saved using **Joblib**, allowing predictions without retraining.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

## Machine Learning Workflow

* Data Loading
* Data Preprocessing
* Label Encoding
* Train-Test Split
* Random Forest Classifier
* Hyperparameter Tuning (RandomizedSearchCV)
* Model Evaluation
* Feature Importance Analysis
* Save & Load Trained Model
* Loan Approval Prediction

---

## Features Used

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

## Target Variable

**loan_status**

* **1** → Loan Approved
* **0** → Loan Rejected

---

## Machine Learning Algorithm

**Random Forest Classifier**

### Hyperparameter Tuning

* RandomizedSearchCV
* 5-Fold Cross Validation

---

## Model Performance

* Test Accuracy: **97.19%**
* Cross Validation Score: **97.98%**

Evaluation Metrics Used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Feature Importance

The Random Forest model identified the following features as the most influential:

1. CIBIL Score
2. Loan Term
3. Loan Amount

CIBIL Score was the most important feature for predicting loan approval.

---

## Visualizations

This project includes:

* Feature Importance
* Confusion Matrix
* Correlation Heatmap
* Loan Status Distribution

---

## Project Structure

```text
Loan-Approval-Prediction/
│
├── Loan_Approval_Prediction.ipynb
├── loan_approval_dataset.csv
├── loan_approval_ml_model.pkl
├── requirements.txt
├── README.md
│
└── images/
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── correlation_heatmap.png
    └── target_distribution.png
```

---

## How to Run

### 1. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 2. Open the Jupyter Notebook

Run all cells to:

* Load the dataset
* Train the model
* Evaluate the model
* Generate visualizations

### 3. Use the Saved Model

Load the trained model using Joblib:

```python
import joblib

model = joblib.load("loan_approval_ml_model.pkl")
```

Then provide new applicant details to predict whether the loan will be **Approved** or **Rejected**.

---

## Future Improvements

* Build a Streamlit web application
* Experiment with XGBoost and LightGBM
* Improve feature engineering
* Deploy the model online

---

## Author

**Aryan Choudhari**

Engineering Student (Artificial Intelligence & Data Science)

Interested in Machine Learning, Data Analytics, and Python Development.
