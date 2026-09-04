# Credit Card Fraud Detection

## 📌 Overview

This project develops a **Machine Learning-based Credit Card Fraud Detection system** to identify potentially fraudulent transactions using transactional and behavioral features.

The project covers the complete ML workflow, including **data preprocessing, exploratory data analysis, feature transformation, model training, hyperparameter tuning, evaluation, model serialization, and API deployment**.

---

## 🎯 Objective

The main objective is to build a reliable classification model capable of distinguishing between **legitimate and fraudulent credit card transactions**.

The dataset contains **10,000 transactions** with features such as:

- Transaction amount
- Transaction hour
- Merchant category
- Foreign transaction indicator
- Location mismatch
- Device trust score
- Transaction velocity
- Cardholder age
- Fraud label

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**
- **FastAPI**
- **Uvicorn**

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning & EDA
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
ColumnTransformer
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Cross-Validation
   ↓
GridSearchCV
   ↓
Best Model Selection
   ↓
Model Evaluation
   ↓
Joblib Serialization
   ↓
FastAPI Deployment
   ↓
Fraud Prediction API
