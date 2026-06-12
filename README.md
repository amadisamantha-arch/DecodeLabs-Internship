# DecodeLabs Internship — Data Science Track
**Intern:** Samantha Amadi | **Batch:** 2026 | **Program:** DecodeLabs Remote Internship

This repository contains my project work for the DecodeLabs 2026 Data Science internship. Each project builds on the last — starting from raw data cleaning all the way to building and evaluating machine learning models. I'm using this as both a learning record and a portfolio of practical work.

---

## Projects

### Project 1 — Advanced EDA & Feature Engineering
**File:** `project1_eda_feature_engineering.py`

The first project was about learning to treat data preparation as a serious engineering task, not just a box to tick before the "real" ML work. I worked with the Titanic dataset and built a full preprocessing pipeline from scratch.

The main things I implemented:

- **Missing value handling** using a decision-matrix approach — dropping rows where missingness was under 5%, using median/group-wise imputation between 5–20%, and KNN imputation for anything above 20%
- **Outlier neutralization** via IQR Winsorization using `numpy.clip()` — I chose capping over deletion because dropping rows destroys other valid features in the same row
- **Feature engineering** — created 5 new predictive features: `family_size`, `is_alone`, `fare_per_person`, `age_group`, and a `pclass_sex` interaction feature
- **One-Hot Encoding** for categorical variables to avoid the false mathematical hierarchy that label encoding creates
- **Collinearity eradication** — built a Pearson correlation matrix and systematically dropped the weaker feature in any pair correlated above 0.80

Everything is written using vectorized Pandas/NumPy operations — no procedural loops anywhere in the transformation logic.

**Dataset:** Titanic (via Seaborn) — 891 rows, 8 raw features → 889 rows, 18 engineered features after cleaning

**Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

---

### Project 2 — Supervised Learning: Fraud Detection Pipeline
**File:** `project2_fraud_detection.py` | **Output:** `fraud_detection_results.png`

This project was about building a classification system for fraud detection — which sounds straightforward until you realize the dataset is 98% legitimate transactions and 2% fraud. A model that just predicts "not fraud" every time gets 98% accuracy and catches nothing. That's the core problem Project 2 is designed to solve.

What I built:

- **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance — SMOTE interpolates new fraud samples rather than just duplicating existing ones, which avoids overfitting
- **Two separate pipelines** using `imblearn.pipeline.Pipeline` (not sklearn's — sklearn's Pipeline can't handle resampling properly):
  - Logistic Regression pipeline: StandardScaler → SMOTE → Classifier
  - Random Forest pipeline: SMOTE → Classifier (no scaler needed, trees are scale-invariant)
- **GridSearchCV with Stratified K-Fold** for hyperparameter tuning — SMOTE runs inside each fold so the validation set is never contaminated with synthetic data
- **Evaluation using Precision, Recall and ROC-AUC** — accuracy is deliberately excluded as a metric here because it's misleading on imbalanced data

**Results:**

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.11 | 0.80 | 0.93 |
| Random Forest | 0.94 | 0.39 | 0.98 |

The two models make an interesting trade-off. Logistic Regression casts a wide net — it catches 80% of fraud but also flags a lot of legitimate transactions. Random Forest is much more precise — when it calls something fraud, it's right 94% of the time, but it misses more cases overall. Which one is "better" depends entirely on the business context.

**Winner by ROC-AUC:** Random Forest (0.9849)

**Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, Seaborn, Matplotlib

---

## How to Run

**Clone the repo:**
```bash
git clone https://github.com/amadisamantha-arch/DecodeLabs-Internship.git
cd DecodeLabs-Internship
```

**Install dependencies:**
```bash
pip install pandas numpy seaborn scikit-learn imbalanced-learn matplotlib
```

**Run a project:**
```bash
python project1_eda_feature_engineering.py
python project2_fraud_detection.py
```

---

## Tech Stack
Python 3 · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Seaborn · Matplotlib

---

*Built as part of the DecodeLabs 2026 Data Science Internship Program.*
