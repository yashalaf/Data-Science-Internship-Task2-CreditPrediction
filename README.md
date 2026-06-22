# 🏦 Credit Risk Prediction Loan Default Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red?logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Task Objective

> **Predict whether a loan applicant is likely to default on a loan** using historical borrower data.

This is a **binary classification** problem:

| Label | Meaning |
|---|---|
| `1` | Loan Paid Back ✅ |
| `0` | Loan Defaulted ❌ |

Banks lose significant revenue from loan defaults. This model helps identify high-risk applicants **before** a loan is issued, enabling smarter lending decisions.

---

## 📂 Repository Structure

```
Credit-Risk-Prediction/
│
├── README.md                          ← Project documentation (this file)
├── Task2_CreditRiskPrediction.ipynb       ← Full Jupyter Notebook (executed)
└── LoanPredictionDataset.csv          ← Dataset (20,000 records)
```

---

## 📦 Dataset

| Property | Detail |
|---|---|
| Name | Loan Prediction Dataset |
| Source | Kaggle |
| Rows | 20,000 |
| Columns | 22 |
| Missing Values | None |
| Class Balance | 80% Paid Back / 20% Default |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age |
| `gender` | Categorical | Male / Female / Other |
| `marital_status` | Categorical | Single / Married / Divorced |
| `education_level` | Categorical | High School / Bachelor / Master / PhD |
| `annual_income` | Numeric | Yearly income in USD |
| `monthly_income` | Numeric | Monthly income in USD |
| `employment_status` | Categorical | Employed / Self-Employed / Unemployed |
| `debt_to_income_ratio` | Numeric | Total debt / Annual income |
| `credit_score` | Numeric | Credit score (300–850) |
| `loan_amount` | Numeric | Requested loan amount in USD |
| `loan_purpose` | Categorical | Reason for loan |
| `interest_rate` | Numeric | Loan interest rate (%) |
| `loan_term` | Numeric | Duration of loan (months) |
| `installment` | Numeric | Monthly payment amount |
| `grade_subgrade` | Categorical | Loan grade (A–G) |
| `num_of_open_accounts` | Numeric | Open credit accounts |
| `total_credit_limit` | Numeric | Total available credit |
| `current_balance` | Numeric | Current debt balance |
| `delinquency_history` | Numeric | Past delinquency count |
| `public_records` | Numeric | Bankruptcy / public records |
| `num_of_delinquencies` | Numeric | Number of delinquencies |
| `loan_paid_back` ⭐ | Binary | **Target** 1 = Paid, 0 = Default |

---

## 🛠️ Approach

### Step 1: Data Cleaning

| Check | Result | Action Taken |
|---|---|---|
| Missing values | ✅ None found | No imputation needed |
| Duplicate rows | ✅ None found | No removal needed |
| Outliers | ⚠️ Present in income, loan amount | IQR Winsorization applied |

Outlier capping applied to: `annual_income`, `loan_amount`, `credit_score`, `debt_to_income_ratio`

### Step 2: Preprocessing

- **Label Encoding** → applied to all 6 categorical columns: `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`
- **StandardScaler** → applied for Logistic Regression (fit on train only, transform on test)
- **Train-Test Split** → 80% train (16,000 samples) / 20% test (4,000 samples), stratified

### Step 3: Exploratory Data Analysis (EDA)

7 visualizations were produced to understand the data before modelling:

| # | Visualization | Insight |
|---|---|---|
| 1 | Target Distribution (Bar + Pie) | 80/20 class split moderate imbalance |
| 2 | Loan Amount (Histogram + Box Plot) | Defaulters borrow slightly higher amounts |
| 3 | Education Level (Count + Default Rate %) | Higher education → lower default rate |
| 4 | Annual Income (Histogram + Violin Plot) | Lower income applicants default more |
| 5 | Credit Score (Histogram + Binned Bars) | Poor score (< 580) → much higher default rate |
| 6 | Correlation Heatmap | `annual_income` & `monthly_income` are highly correlated |
| 7 | Employment / Loan Purpose / Gender | Unemployed applicants have highest default rate |

### Step 4: Model Training

Two classification models were trained and compared:

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `C=1.0`, `max_iter=1000`, `random_state=42` — trained on scaled features |
| **Decision Tree** | `max_depth=6`, `min_samples_split=50`, `min_samples_leaf=20`, `random_state=42` |

---

## 📊 Evaluation

### Performance Metrics (Test Set: 4,000 samples)

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Train Accuracy | 88.70% | 90.08% |
| **Test Accuracy** | 88.60% | **89.85%** |
| Precision | 0.8979 | 0.8933 |
| Recall | 0.9675 | **0.9916** |
| F1-Score | 0.9314 | **0.9399** |
| **ROC-AUC** | 0.8521 | **0.8725** |

### Confusion Matrix

| | Logistic Regression | Decision Tree |
|---|---|---|
| True Negatives (TN) | 448 | 421 |
| False Positives (FP) | 352 | 379 |
| False Negatives (FN) | 104 | **27** |
| True Positives (TP) | 3,096 | **3,173** |

> Decision Tree has only **27 False Negatives** vs 104 for Logistic Regression meaning it correctly identifies far more actual defaulters, which is the most critical outcome for a bank.

### Feature Importances (Decision Tree)

| Rank | Feature | Importance Score |
|---|---|---|
| 🥇 1 | `employment_status` | 0.6655 |
| 🥈 2 | `credit_score` | 0.1646 |
| 🥉 3 | `debt_to_income_ratio` | 0.1596 |
| 4 | `grade_subgrade` | 0.0035 |
| 5 | `education_level` | 0.0016 |

---

## 💡 Results & Insights

### Model Verdict
**Decision Tree outperforms Logistic Regression** across all key metrics, with higher accuracy (89.85% vs 88.60%), better F1-Score (0.9399 vs 0.9314), and higher ROC-AUC (0.8725 vs 0.8521).

### Recommendations for Improvement

| Problem | Solution |
|---|---|
| Class imbalance (80/20) | Apply SMOTE or `class_weight='balanced'` |
| Only 2 models tested | Try Random Forest or XGBoost |
| No cross-validation | Use k-fold CV for more reliable estimates |
| Label Encoding on nominal features | Try One-Hot Encoding for `gender`, `loan_purpose` |

---

## 🧰 Technologies Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.0+ | Data loading, cleaning, manipulation |
| `numpy` | 1.24+ | Numerical operations, array handling |
| `matplotlib` | 3.7+ | Custom plots, figure layout |
| `seaborn` | 0.12+ | Statistical visualizations (violin, heatmap) |
| `scikit-learn` | 1.3+ | ML models, preprocessing, metrics |
| `jupyter` | 7.0+ | Interactive notebook environment |

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Data-Science-Internship-Task2-CreditPrediction.git
cd Credit-Risk-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook Task2_CreditRiskPrediction.ipynb
```

### 4. Run all cells
Go to: **Kernel → Restart & Run All**

> ⚠️ Make sure `LoanPredictionDataset.csv` is in the same folder as the notebook before running.

---

## 🔑 Skills Demonstrated

- ✅ Data cleaning and handling missing values
- ✅ Outlier detection and treatment (IQR Winsorization)
- ✅ Exploratory Data Analysis (EDA) with 7 visualizations
- ✅ Feature encoding (Label Encoding) and scaling (StandardScaler)
- ✅ Binary classification using Logistic Regression and Decision Tree
- ✅ Model evaluation: accuracy, confusion matrix, precision, recall, F1-score, ROC-AUC
- ✅ Feature importance analysis and business insight generation

---

*Task 2 Data Science Internship
