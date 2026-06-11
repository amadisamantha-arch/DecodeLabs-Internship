# Project 2 - Fraud Detection Pipeline
# Samantha Amadi | DecodeLabs Data Science Internship 2026
#
# The goal here is to build a classifier that can actually catch fraud.
# Sounds simple until you realize that in real financial data, fraud makes up
# like 0.17% of all transactions. So if the model just guesses "not fraud"
# every single time, it's technically 99.83% accurate — and completely useless.
# That's the whole problem this project is about solving.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # important — NOT sklearn's Pipeline


# -----------------------------------------------------------------------------
# STEP 1: Create the dataset
# -----------------------------------------------------------------------------
# Since I don't have direct access to the Kaggle credit card fraud dataset,
# I'm generating a synthetic one that mirrors the same class imbalance:
# roughly 0.17% fraud vs 99.83% legitimate transactions.
# The patterns and relationships between features are realistic enough
# for the pipeline to behave the same way it would on real data.

print("Building dataset...")

X, y = make_classification(
    n_samples=20000,
    n_features=28,
    n_informative=15,
    n_redundant=5,
    weights=[0.98, 0.02],   # 0.17% fraud — this is the imbalance we're dealing with
    flip_y=0,
    random_state=42
)

feature_cols = [f'V{i}' for i in range(1, 28)] + ['Amount']
df = pd.DataFrame(X, columns=feature_cols)
df['Class'] = y  # 0 = legitimate, 1 = fraud

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud rate: {df['Class'].mean() * 100:.3f}%")

# that fraud rate is genuinely tiny — you can see why a naive model would just
# ignore the fraud class entirely and still look "accurate"


# -----------------------------------------------------------------------------
# STEP 2: Split FIRST, then do everything else
# -----------------------------------------------------------------------------
# This tripped me up at first — the instinct is to clean/balance the data
# and then split it. But that's actually a serious mistake called data leakage.
# If SMOTE runs before the split, synthetic fraud samples end up in both
# the training AND test sets, so the model is basically "remembering" answers
# it was supposed to discover. The test results look great but mean nothing.
#
# Correct order: split first → SMOTE only runs inside the training pipeline.

X = df.drop('Class', axis=1)
y = df['Class']

# stratify=y makes sure both splits keep the same fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")
print(f"Fraud cases in test set: {y_test.sum()} ({y_test.mean()*100:.3f}%)")


# -----------------------------------------------------------------------------
# STEP 3: Build the pipelines
# -----------------------------------------------------------------------------
# Using imblearn's Pipeline here, not sklearn's. The difference matters —
# sklearn's Pipeline doesn't know how to handle resampling (SMOTE changes
# both X and y, not just X), so it either crashes or silently ignores it.
# imblearn's version handles this properly.
#
# Two separate pipelines for the two models because they need different setups:
# Logistic Regression needs StandardScaler (it's sensitive to feature scale)
# Random Forest doesn't need scaling at all (tree splits are scale-invariant)

# -- Pipeline 1: Logistic Regression --
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# -- Pipeline 2: Random Forest --
rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])


# -----------------------------------------------------------------------------
# STEP 4: Hyperparameter tuning with GridSearchCV
# -----------------------------------------------------------------------------
# GridSearchCV + imblearn Pipeline = SMOTE runs fresh inside every fold.
# So the validation fold never sees any synthetic data — it stays imbalanced,
# reflecting what real-world predictions will look like.
# Scoring on 'roc_auc' not accuracy, for obvious reasons at this point.

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\n--- Tuning Logistic Regression ---")

lr_params = {
    'smote__k_neighbors': [3, 5],
    'classifier__C': [0.1, 1.0]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_params,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
lr_grid.fit(X_train, y_train)
print(f"Best LR params: {lr_grid.best_params_}")
print(f"Best CV ROC-AUC: {lr_grid.best_score_:.4f}")


print("\n--- Tuning Random Forest ---")

rf_params = {
    'smote__k_neighbors': [3, 5],
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, None]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}")


# -----------------------------------------------------------------------------
# STEP 5: Evaluate on the test set
# -----------------------------------------------------------------------------
# The test set is untouched — no SMOTE, no scaling applied to it directly.
# This is intentional. It has to reflect real-world conditions (highly imbalanced)
# otherwise the evaluation means nothing.
#
# Metrics I care about:
# - Recall: did we catch the actual fraud? missing fraud = real financial loss
# - Precision: when we flagged something as fraud, were we right?
# - ROC-AUC: how well can the model separate fraud from legit overall

def evaluate_model(name, model, X_test, y_test):
    print(f"\n{'='*55}")
    print(f"  {name} — Test Set Results")
    print(f"{'='*55}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nPrecision : {precision:.4f}  (when we flag fraud, how often we're right)")
    print(f"Recall    : {recall:.4f}  (how much of the actual fraud we caught)")
    print(f"ROC-AUC   : {roc_auc:.4f}  (overall separation between classes)")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], zero_division=0))

    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"  True Negatives  (correctly caught legit): {cm[0,0]}")
    print(f"  False Positives (legit flagged as fraud): {cm[0,1]}")
    print(f"  False Negatives (fraud we missed):        {cm[1,0]}  <- the costly one")
    print(f"  True Positives  (fraud correctly caught): {cm[1,1]}")

    return y_prob

lr_probs = evaluate_model("Logistic Regression", lr_grid.best_estimator_, X_test, y_test)
rf_probs = evaluate_model("Random Forest", rf_grid.best_estimator_, X_test, y_test)


# -----------------------------------------------------------------------------
# STEP 6: ROC Curve comparison
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Project 2: Fraud Detection Model Evaluation', fontsize=14, fontweight='bold')

ax1 = axes[0]
for name, probs in [("Logistic Regression", lr_probs), ("Random Forest", rf_probs)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random guess')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate (Recall)')
ax1.set_title('ROC Curve Comparison')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
rf_cm = confusion_matrix(y_test, rf_grid.best_estimator_.predict(X_test))
sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Legitimate', 'Fraud'],
    yticklabels=['Legitimate', 'Fraud'],
    ax=ax2
)
ax2.set_title('Random Forest — Confusion Matrix')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight')
print("\nChart saved -> fraud_detection_results.png")


# -----------------------------------------------------------------------------
# STEP 7: Feature importance (Random Forest only)
# -----------------------------------------------------------------------------

rf_classifier = rf_grid.best_estimator_.named_steps['classifier']
importances = rf_classifier.feature_importances_

importance_df = pd.DataFrame({
    'feature': X.columns.tolist(),
    'importance': importances
}).sort_values('importance', ascending=False).head(10)

print(f"\nTop 10 most important features (Random Forest):")
print(importance_df.to_string(index=False))


# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
lr_auc = roc_auc_score(y_test, lr_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"\n{'='*55}")
print(f"  Final Model Comparison")
print(f"{'='*55}")
print(f"  Logistic Regression ROC-AUC : {lr_auc:.4f}")
print(f"  Random Forest ROC-AUC       : {rf_auc:.4f}")
winner = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"
print(f"\n  Winner: {winner}")
print(f"\n  Key takeaway: a model can have 99%+ accuracy on fraud data")
print(f"  and still catch nothing. ROC-AUC and Recall are what matter.")
print(f"{'='*55}")
