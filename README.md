# 🧪 DecodeLabs Internship — Data Science Projects
**Intern:** Samantha Amadi | **Batch:** 2026 | **Track:** Data Science

---

## 📁 Project 1: Advanced EDA & Feature Engineering

### Overview
This project transforms a raw, chaotic dataset into a mathematically clean, ML-ready feature store using the **Input-Process-Output (IPO) pipeline architecture**. The focus is on statistical precision over arbitrary guesswork — every transformation is rule-based and mathematically justified.

**Dataset used:** Titanic (891 passengers, 8 raw features → 18 engineered features)

---

### ⚙️ What the Script Does

#### Phase 1 — Securing Input Fidelity
- Calculates missingness proportion per feature
- Applies the **Missing Data Decision Matrix:**
  - `< 5%` missing → Row deletion (`dropna`) to prevent synthetic bias
  - `5–20%` missing → Statistical imputation (Global Median for skewed numeric; Group-wise Mode for categorical)
  - `> 20%` missing → KNN Imputation (`k=5`) for multi-dimensional estimation
- Detects and neutralizes outliers using **IQR Winsorization** (`numpy.clip()`) — values are capped at `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR`, preserving row count and sequential integrity

#### Phase 2 — Vectorized Feature Engineering
Five new predictive features engineered using pure Pandas/NumPy (no procedural loops):

| Feature | Description | Rationale |
|---|---|---|
| `family_size` | `sibsp + parch + 1` | Group travel affected survival odds |
| `is_alone` | Binary flag if travelling solo | Solo passengers behaved differently |
| `fare_per_person` | `fare / family_size` | More accurate socioeconomic proxy |
| `age_group` | Binned: Child / Teen / YoungAdult / Adult / Senior | Survival varied by life stage |
| `pclass_sex` | Class + Sex interaction feature | Women in 1st class had ~100% survival |

- **One-Hot Encoding** applied to all nominal categorical columns (`drop_first=True` to avoid dummy variable trap)
- **Collinearity Eradication** — Pearson correlation matrix built; pairs above `r = 0.80` resolved by dropping the feature with weaker correlation to the target variable

#### Phase 3 — Output
- Final dataset: **889 rows × 18 features**, zero missing values
- Exported to `titanic_cleaned_features.csv` — ready for downstream ML estimators

---

### 🛠️ Tech Stack
- Python 3
- Pandas
- NumPy
- Scikit-learn (KNNImputer)
- Seaborn (dataset loading)
- Matplotlib

---

### 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/amadisamantha-arch/DecodeLabs-Internship.git
cd DecodeLabs-Internship
```

**2. Install dependencies**
```bash
pip install pandas numpy seaborn scikit-learn matplotlib
```

**3. Run the script**
```bash
python project1_eda_feature_engineering.py
```

The cleaned dataset will be saved as `titanic_cleaned_features.csv` in the same directory.

---

### 📊 Key Results

| Metric | Value |
|---|---|
| Raw shape | 891 × 8 |
| Final shape | 889 × 18 |
| Missing values remaining | 0 |
| Outliers neutralized | 438 values capped |
| New features engineered | 5 |
| Collinear features removed | 3 |

---

*Built as part of the DecodeLabs 2026 Data Science Internship Program.*
