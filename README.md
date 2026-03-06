# 🏠 House Price Prediction — Kaggle Competition

> **Goal:** Predict the sale price of houses in Ames, Iowa based on 80 features describing each property.
> **Competition:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 📌 Project Overview

This project is part of a structured 10-phase ML/DL/AI learning roadmap. The House Prices dataset serves as the continuous thread across Phases 1–5, with each phase adding a new layer to the same pipeline:

| Phase | What was done |
|---|---|
| Phase 1 + 2 | EDA + Full Preprocessing Pipeline ← current |
| Phase 3 | Train & compare ML models |
| Phase 4 | Cross-validation + regularization |
| Phase 5 | Evaluation metrics + SHAP + Kaggle submission |

---

## 📂 Dataset

The dataset comes from the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and contains:

| File | Description |
|---|---|
| `train.csv` | 1,460 houses with all features + SalePrice |
| `test.csv` | 1,459 houses with all features, no SalePrice |
| `sample_submission.csv` | Example of expected Kaggle submission format |
| `data_description.txt` | Description of every column and its possible values |

**Target variable:** `SalePrice` — the sale price of each house in USD

---

## 🧹 Phase 1+2 — EDA & Preprocessing

### What this notebook covers:

**1. Exploratory Data Analysis**
- Shape, dtypes, numeric vs categorical column counts
- Missing values summary — count, percentage, and type per column
- Domain understanding: distinguishing true missing data from feature absence (e.g. `PoolQC = NA` means no pool, not missing data)

**2. Missing Value Treatment**
- High-missing columns where NA = absence → filled with `"None"` or `0`
- Numerical columns → median or neighborhood-grouped median imputation
- Categorical columns → mode imputation
- All statistics computed from train only, then applied to test

**3. Outlier Detection & Treatment**
- IQR-based detection on all numeric columns
- Manual inspection of suspicious values (cross-checked with SalePrice)
- Blind winsorization applied with IQR > 0 guard to avoid zero-variance columns

**4. Scaling**
- `StandardScaler` applied to all numeric columns except `SalePrice`

**5. Encoding**
- Nominal columns (22) → Target Encoding (`category_encoders`)
- Ordinal columns (21) → Label Encoding (to be improved with OrdinalEncoder in Phase 5)

**6. Feature Selection**
- Correlation with SalePrice computed for all columns
- No features dropped — all within acceptable range

---

## ⚠️ Known Limitations & Future Improvements

- **Blind winsorization** was applied for time efficiency — in production, each column should be inspected individually
- **LabelEncoder** assigns numbers alphabetically, not by true ordinal order (Po < Fa < TA < Gd < Ex) — will be revisited with `OrdinalEncoder` if model performance is poor
- **Target encoding** can cause data leakage — will be handled properly with cross-validation in Phase 3+

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Scaling, encoding, pipeline |
| category_encoders | Target encoding |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/adam1861/House-Price-Prediction.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders

# Open the notebook
jupyter notebook Preprocessing.ipynb
```

> ⚠️ Update the file paths in cells 3 and 5 to point to your local dataset files.

---

## 📈 Results

> Model training and evaluation coming in Phases 3–5.
> Kaggle submission score will be added here after Phase 5.

---

## 👤 Author

**Adam** — Learning ML/DL/AI through a structured 10-phase roadmap.
Following the philosophy: *understand deeply, not rush through.*
