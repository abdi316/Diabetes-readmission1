# Predicting Early Readmission of Diabetic Patients Using Machine Learning

**Authors:** Abdifatah Hussein · Isihack Harerimana · Shamel Iqbal  
**Course:** DATA 4382 — Spring 2026  
**Instructor:** Professor Rostami

---

## 1. Motivation

Diabetes is a chronic, incurable condition and one of the leading drivers of hospital readmission costs in the United States. Patients readmitted within 30 days represent a major financial and clinical burden — for both patients and healthcare systems — yet traditional risk tools like the **LACE index** perform poorly on this population. They fail to capture subtle interactions between demographic, clinical, and behavioral features that separate high-risk patients from low-risk ones.

This project investigates whether machine learning can do better. By training on 100k+ real patient encounters, the goal is to build a model that flags at-risk diabetic patients before discharge — enabling earlier intervention, better resource allocation, and improved patient outcomes.

---

## 2. Project Overview

We frame readmission as a **binary classification task** (readmitted within 30 days: yes / no), apply a structured preprocessing pipeline, and evaluate three model families — Random Forest, XGBoost, and Multi-Layer Perceptron — across three class-balancing strategies.

| Metric | Value |
|---|---|
| Dataset size | 101,766 patient encounters |
| Features | 50 |
| Target | Binary — readmitted < 30 days |
| Best ROC-AUC | 0.65 (Random Forest + undersampling) |
| Best Recall | 62% of readmissions caught |


<img width="1923" height="765" alt="01_class_distribution" src="https://github.com/user-attachments/assets/f296b901-eece-4236-87a9-867bbe848e49" />



---

## 3. Data

- **Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Type:** Tabular — categorical, numerical, and ordinal features
- **Size:** 101,766 records · 50 features · 130 U.S. hospitals
- **Key features:** Race, gender, age · Primary/secondary diagnoses · Emergency, inpatient, outpatient visit counts · Diabetes medications and dosage changes · Lab results (HbA1c, max glucose serum)
- **Class imbalance:** 88% not readmitted early · 12% readmitted within 30 days

---

## 4. Data Preprocessing

The preprocessing pipeline was designed for reproducibility — any new dataset with the same schema can be passed through without manual adjustments.

**Pipeline steps:**

```
Raw Data → Data Cleaning → Missing Value Imputation → Feature Engineering → Encoding → 80/20 Split
```

1. **Initial cleaning** — dropped rows missing >30% of attributes; removed records with all three diagnosis fields empty or unknown gender values
2. **Column filtering** — removed columns missing >45% of values; dropped `encounter_id`, `patient_nbr`, `payer_code`; removed near-constant columns (>90% single value)
3. **Missing value imputation** — median for numerical features; mode for categorical features
4. **Rare category merging** — categories with <5% frequency collapsed into `other`
5. **Feature engineering** — created `visits_sum`, `number_medicaments`, `number_medicaments_changes`; ICD-9 codes mapped to clinical categories
6. **Ordinal encoding** — age decade-ranges mapped to integers 0–9
7. **One-hot encoding** — all remaining categorical features encoded
8. **Normalization / outlier removal** — determined unnecessary after analysis

---

## 5. Exploratory Data Analysis (EDA)

### Class Distribution

The original dataset contains three target classes that are collapsed into a binary label:

| Class | Count | Share |
|---|---|---|
| Not readmitted | ~56,000 | 55% |
| Readmitted > 30 days | ~35,500 | 35% |
| Readmitted < 30 days (positive class) | ~10,200 | 10% |

After binarization: **88% negative / 12% positive** — a heavily imbalanced dataset.

### Key EDA Insights

- **Diagnosis code 428** (heart failure) appears most frequently among readmitted patients — circulatory comorbidities are a strong signal
- **Number of prior inpatient visits** and **total medication count** are the second and third strongest predictors
- The severe class imbalance (88/12) makes accuracy a misleading metric — all evaluation uses ROC-AUC and recall
- Model performance degrades with patient age, suggesting complex clinical patterns in elderly patients that the current feature set cannot fully capture

### Top Predictive Features (from EDA)

```
<img width="1625" height="1021" alt="06_feature_importance (1)" src="https://github.com/user-attachments/assets/d214f9a9-a907-4dbb-8bbc-ebcdfe6227d6" />

```

---

## 6. Modeling Approach

| Model | Type | Why selected |
|---|---|---|
| **Random Forest** (baseline + advanced) | Ensemble (bagging) | Handles high-dimensional tabular data; provides feature importance; robust via tree averaging |
| **XGBoost** | Ensemble (boosting) | Sequentially corrects errors; strong on structured data; efficient with mixed feature types |
| **Multi-Layer Perceptron** | Neural network | Captures non-linear interactions; benchmark for whether deep patterns exist in this dataset |

Each model was trained on three dataset versions: **original** (imbalanced), **undersampling**, and **SMOTE oversampling** — to evaluate how class balancing affects prediction.

<img width="1923" height="874" alt="07_model_comparison_auc" src="https://github.com/user-attachments/assets/22005e6f-a8cc-42e6-a5ae-f16d55475351" />



---

## 7. Model Training

- **Tools:** `scikit-learn`, `xgboost`, `pandas`, `numpy` · Google Colab / Jupyter Notebook
- **Train / test split:** 80% training · 20% test · stratified on target label
- **Cross-validation:** Applied to ensure robustness and reduce performance variance
- **Class balancing strategies:** Undersampling · SMOTE oversampling · `class_weight='balanced'`
- **Key hyperparameters (Random Forest):** `max_depth=8` · `random_state=42` · `class_weight='balanced'`

---

## 8. Results

> **Why ROC-AUC?** Given the class imbalance, accuracy is misleading (predicting all-negative gives 88% accuracy). ROC-AUC measures the model's ability to rank positive cases above negative ones regardless of threshold. **Recall** is the secondary metric because missing a high-risk patient is clinically more costly than a false alarm.

### Model Comparison Table

| Model | Sampling | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **Random Forest** | Undersampling | **0.65** | **62%** | ~20% | ~0.30 |
| Random Forest | Class weights | 0.63 | 58% | ~19% | ~0.28 |
| XGBoost | Undersampling | 0.63 | 55% | ~19% | ~0.27 |
| MLP | Undersampling | 0.60 | 50% | ~18% | ~0.25 |
| Random Forest | Oversampling | 0.58 | 48% | ~17% | ~0.24 |

### Key Results

- Random Forest with undersampling achieved **ROC-AUC = 0.65** and **recall = 62%**
- False positive rate of ~40% is an acceptable trade-off — catching more high-risk patients allows earlier hospital intervention
- **Sampling strategy mattered more than model architecture** — undersampling consistently outperformed oversampling across all three models
- Neural networks did not outperform Random Forest on this structured tabular dataset

<img width="1473" height="721" alt="08_rfecv_feature_selection" src="https://github.com/user-attachments/assets/650d29f6-cf7b-4b9e-a61a-f57051a8f361" />


---

## 9. Model Interpretation

### Feature Importance (Best Random Forest Model)

```
<img width="1923" height="737" alt="03_age_distribution" src="https://github.com/user-attachments/assets/a1bf5ecc-1846-48a2-a15e-b11434685b97" />

```

### What Drives Predictions

- **`num_lab_procedures`** — patients undergoing more lab tests tend to be sicker and at higher readmission risk
- **`num_medications`** — complex medication regimens signal multi-morbidity and systemic fragility
- **`number_inpatient`** — prior inpatient visits are the strongest clinical history signal; frequent hospitalization predicts future hospitalization
- **`visits_sum`** (engineered) — total healthcare utilization across all visit types is more predictive than any single visit type alone
- **Circulatory diagnosis** — patients with a primary circulatory diagnosis (ICD-9 code 428: heart failure) show disproportionately high readmission rates

> 
---

## 10. Key Insights

- **Sampling strategy > model complexity.** Undersampling outperformed oversampling across all three models. SMOTE oversampling degraded performance, likely by distorting the decision boundary.
- **Simpler models outperformed neural networks.** Random Forest beat MLP on this structured tabular dataset — a consistent finding in clinical ML literature on structured EHR data.
- **Older patients are harder to model.** Age-stratified experiments confirmed that prediction accuracy decreases with age, suggesting more complex clinical trajectories that the current feature set cannot fully capture.
- **Clinical impact of 62% recall.** Even at a 40% false positive rate, flagging 62% of high-risk patients gives healthcare providers an actionable list for targeted follow-up — reducing avoidable readmissions and their associated financial and human costs.

---

## 11. Conclusion

Machine learning can provide meaningful early readmission risk signals for diabetic patients, surpassing the limitations of traditional rule-based tools. The best-performing model — **Random Forest with undersampling** — achieved a ROC-AUC of 0.65 and correctly identified 62% of patients who would be readmitted within 30 days. Results were more sensitive to class-balancing strategy than to model architecture, and feature importance analysis confirmed that clinical utilization metrics (lab procedures, medications, prior visits) are the most predictive signals in this dataset.


---

## 13. How to Run

```bash
git clone https://github.com/your-repo/diabetes-readmission
cd diabetes-readmission
pip install -r requirements.txt
```

```bash
# Step 1: Preprocess data
jupyter nbconvert --to notebook --execute notebooks/1_preprocessing.ipynb

# Step 2: Exploratory data analysis
jupyter nbconvert --to notebook --execute notebooks/2_eda.ipynb

# Step 3: Train models
jupyter nbconvert --to notebook --execute notebooks/3_modeling.ipynb

# Step 4: Complementary experiments
jupyter nbconvert --to notebook --execute notebooks/4_complementary_experiments.ipynb
```

---

## 14. Repository Structure

```
diabetes-readmission/
│
├── notebooks/
│   ├── 1_preprocessing.ipynb         # Cleaning, imputation, feature engineering, pipeline
│   ├── 2_eda.ipynb                   # EDA — class distributions, feature importance, diagnosis breakdowns
│   ├── 3_modeling.ipynb              # Baseline and advanced model training (RF, XGBoost, MLP)
│   └── 4_complementary_experiments.ipynb  # Age-stratified and cluster-based ensemble experiments
│
├── data/
│   ├── X_train.csv                   # Preprocessed training features (output of notebook 1)
│   ├── X_test.csv                    # Preprocessed test features
│   ├── y_train.csv                   # Binary readmission labels — training set
│   └── y_test.csv                    # Binary readmission labels — test set
│
├── src/
│   ├── preprocessing/
│   │   ├── transformers.py           # Custom sklearn-compatible transformer classes
│   │   └── helpers.py                # Utility functions: load_dataset(), transform_label()
│   └── evaluation.py                 # evaluate_model(), roc_auc(), undersample(), CombinedModel
│
├── requirements.txt                  # All Python dependencies with pinned versions
└── README.md                         # This file
```

---

## 15. Requirements

```bash
pip install -r requirements.txt
```

```
pandas==2.1.0
numpy==1.25.2
scikit-learn==1.3.0
xgboost==2.0.0
matplotlib==3.8.0
seaborn==0.13.0
imbalanced-learn==0.11.0
jupyter==1.0.0
```
