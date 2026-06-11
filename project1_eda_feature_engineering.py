# =============================================================================
# DECODELABS INTERNSHIP - DATA SCIENCE PROJECT 1
# Advanced EDA & Feature Engineering
# Author: Samantha Amadi
# Dataset: Titanic (via Seaborn)
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# =============================================================================
# PHASE 0: LOAD DATASET
# =============================================================================

df = sns.load_dataset('titanic')

# Keep only the columns relevant to our pipeline
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

print("=" * 60)
print("PHASE 0: RAW DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)


# =============================================================================
# PHASE 1: SECURING INPUT FIDELITY
# Missing Value Analysis + Statistical Imputation
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 1: SECURING INPUT FIDELITY")
print("=" * 60)

# ── Step 1: Calculate missingness proportion per feature ─────────────────────
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_report = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_pct.round(2)
}).sort_values('Missing %', ascending=False)

print("\nMissingness Report:")
print(missing_report[missing_report['Missing Count'] > 0])

# ── Step 2: Apply the Missing Data Decision Matrix ───────────────────────────
#
#   < 5%   → Drop rows (dropna) — preserves volume, prevents synthetic bias
#   5-20%  → Statistical Imputation
#              Skewed numeric   → Global Median
#              Categorical      → Sub-Group Conditional Imputation
#   > 20%  → Multi-Dimensional Estimation → KNN Imputation
#

for col in df.columns:
    pct = missing_pct[col]

    if pct == 0:
        continue  # No missing values, skip

    elif pct < 5:
        # Drop rows — missingness too small to justify synthetic replacement
        before = len(df)
        df = df.dropna(subset=[col])
        print(f"\n[<5%] '{col}' ({pct:.2f}% missing) → Dropped {before - len(df)} rows")

    elif 5 <= pct <= 20:
        if df[col].dtype in ['float64', 'int64']:
            # Skewed numeric → Global Median (robust against outliers)
            skewness = df[col].skew()
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"\n[5-20%] '{col}' ({pct:.2f}% missing, skew={skewness:.2f}) → Filled with Median: {median_val}")
        else:
            # Categorical/Correlated → Sub-Group Conditional Imputation
            # Fill with the most frequent value per 'pclass' group
            df[col] = df.groupby('pclass')[col].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
            )
            print(f"\n[5-20%] '{col}' ({pct:.2f}% missing) → Filled with group-wise mode (by pclass)")

    else:
        # > 20% → KNN Imputation (captures multi-dimensional relationships)
        # KNN only works on numeric data, so we encode first
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        knn_imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        print(f"\n[>20%] '{col}' ({pct:.2f}% missing) → KNN Imputation applied (k=5)")

print(f"\nMissing values after imputation: {df.isnull().sum().sum()}")
print(f"Dataset shape after imputation: {df.shape}")


# =============================================================================
# PHASE 1 (CONT.): OUTLIER DETECTION & NEUTRALIZATION
# IQR-Based Winsorization using numpy.clip()
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 1 (CONT.): OUTLIER DETECTION & NEUTRALIZATION")
print("=" * 60)

# Apply IQR Winsorization to all numeric columns
# Formula: Lower Bound = Q1 - 1.5 * IQR | Upper Bound = Q3 + 1.5 * IQR
# numpy.clip() caps values at the boundary → preserves row count & sequence

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude binary target column 'survived' from outlier treatment
numeric_cols = [c for c in numeric_cols if c != 'survived']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers before capping
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    # Winsorize: cap values at the IQR boundaries (no rows dropped)
    df[col] = np.clip(df[col], lower_bound, upper_bound)

    print(f"'{col}' → Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] | Outliers capped: {outlier_count}")


# =============================================================================
# PHASE 2: VECTORIZED FEATURE ENGINEERING
# Engineer at least 3 new predictive features
# No procedural for-loops — pure Pandas/NumPy vectorized operations
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2: FEATURE ENGINEERING (3+ NEW FEATURES)")
print("=" * 60)

# ── Feature 1: Family Size ───────────────────────────────────────────────────
# Rationale: Passengers traveling with family had different survival patterns.
# Combines SibSp (siblings/spouses) and Parch (parents/children) + self
df['family_size'] = df['sibsp'] + df['parch'] + 1
print(f"\n[Feature 1] 'family_size' created → Range: {df['family_size'].min()} to {df['family_size'].max()}")

# ── Feature 2: Is Alone ──────────────────────────────────────────────────────
# Rationale: Solo travelers behaved differently from group travelers.
# Binary flag derived from family_size using vectorized np.where()
df['is_alone'] = np.where(df['family_size'] == 1, 1, 0)
alone_count = df['is_alone'].sum()
print(f"[Feature 2] 'is_alone' created → {alone_count} solo passengers ({alone_count/len(df)*100:.1f}%)")

# ── Feature 3: Fare Per Person ───────────────────────────────────────────────
# Rationale: Raw fare may be shared among a group. Per-person fare is a
# more accurate proxy for socioeconomic status.
df['fare_per_person'] = df['fare'] / df['family_size']
print(f"[Feature 3] 'fare_per_person' created → Mean: £{df['fare_per_person'].mean():.2f}")

# ── Feature 4 (Bonus): Age Group ─────────────────────────────────────────────
# Rationale: Survival rates varied significantly by life stage.
# Uses vectorized pd.cut() — no loops
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
)
print(f"[Feature 4] 'age_group' created →\n{df['age_group'].value_counts()}")

# ── Feature 5 (Bonus): Pclass-Sex Interaction ────────────────────────────────
# Rationale: The combination of class and sex was a strong survival predictor
# (women in 1st class had near 100% survival). Interaction feature captures this.
df['pclass_sex'] = df['pclass'].astype(str) + '_' + df['sex']
print(f"\n[Feature 5] 'pclass_sex' interaction created →\n{df['pclass_sex'].value_counts()}")


# =============================================================================
# PHASE 2 (CONT.): CATEGORICAL ENCODING
# One-Hot Encoding — avoids false mathematical distance from Label Encoding
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2 (CONT.): CATEGORICAL ENCODING")
print("=" * 60)

# One-Hot Encode nominal categorical columns
# drop_first=True eliminates the dummy variable trap (multicollinearity)
cols_to_encode = ['sex', 'embarked', 'age_group', 'pclass_sex']
df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

print(f"Shape before encoding: {df.shape}")
print(f"Shape after encoding:  {df_encoded.shape}")
print(f"\nNew columns added: {[c for c in df_encoded.columns if c not in df.columns]}")


# =============================================================================
# PHASE 2 (CONT.): COLLINEARITY ERADICATION
# Remove features with Pearson correlation > 0.80 (keep the one closer to target)
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2 (CONT.): COLLINEARITY ERADICATION")
print("=" * 60)

# Work only with numeric columns for correlation matrix
numeric_df = df_encoded.select_dtypes(include=np.number)

# Step 1: Build absolute correlation matrix
corr_matrix = numeric_df.corr().abs()

# Step 2: Isolate upper triangle (avoid duplicate pairs)
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Step 3: Identify pairs with correlation > 0.80
high_corr_pairs = [
    (col, row, upper_triangle.loc[row, col])
    for col in upper_triangle.columns
    for row in upper_triangle.index
    if upper_triangle.loc[row, col] > 0.80
]

if high_corr_pairs:
    print(f"\nHighly correlated pairs found (>0.80):")
    cols_to_drop = set()
    for col_a, col_b, corr_val in high_corr_pairs:
        # Step 4: Drop the feature with weaker correlation to target 'survived'
        corr_a = abs(numeric_df[col_a].corr(numeric_df['survived']))
        corr_b = abs(numeric_df[col_b].corr(numeric_df['survived']))
        weaker = col_a if corr_a < corr_b else col_b
        cols_to_drop.add(weaker)
        print(f"  {col_a} ↔ {col_b}: r={corr_val:.2f} → Dropping '{weaker}'")

    df_encoded = df_encoded.drop(columns=list(cols_to_drop), errors='ignore')
    print(f"\nDropped {len(cols_to_drop)} collinear feature(s): {cols_to_drop}")
else:
    print("No highly correlated pairs found (threshold: 0.80). Dataset is clean.")


# =============================================================================
# PHASE 3: FINAL CLEAN DATASET SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 3: FINAL CLEAN DATASET SUMMARY")
print("=" * 60)

print(f"\nFinal shape: {df_encoded.shape}")
print(f"Missing values remaining: {df_encoded.isnull().sum().sum()}")
print(f"\nFinal columns ({len(df_encoded.columns)}):")
for col in df_encoded.columns:
    print(f"  - {col} ({df_encoded[col].dtype})")

print("\nFinal dataset preview:")
print(df_encoded.head())

# Save cleaned dataset to CSV
df_encoded.to_csv('titanic_cleaned_features.csv', index=False)
print("\n✅ Clean dataset saved to: titanic_cleaned_features.csv")
print("✅ Project 1 Complete — Ready for Machine Learning!")
