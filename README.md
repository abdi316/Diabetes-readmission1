# Diabetes-readmission-prediction

Predicting Early Readmission of Diabetic Patients
Overview

This project predicts 30 day hospital readmissions for diabetic patients using machine learning. The goal is to identify high risk patients early to improve outcomes and reduce healthcare costs.

Dataset
Source: UCI Machine Learning Repository
Size: ~101,766 patient records across 130 U.S. hospitals
Features: Demographics, diagnoses, medications, hospital visits
Target: Binary classification (readmitted within 30 days vs not)
Methodology
Data cleaning, feature engineering, and one-hot encoding
Handled class imbalance using undersampling
Train/test split: 80/20

Models used:

Random Forest
XGBoost
Neural Network (MLP)
Results
Best model: Random Forest
ROC-AUC: ~0.65
Recall: ~60%

Findings show moderate performance, with effectiveness influenced by class imbalance and feature limitations.

Key Insights
Prior hospital visits are strong predictors
Undersampling improved model performance
Tree-based models outperformed neural networks
Tech Stack

Python, Scikit-learn, XGBoost, Pandas, NumPy

Authors

Abdifatah Hussein
Isihack Harerimana
Shamel Iqbal
