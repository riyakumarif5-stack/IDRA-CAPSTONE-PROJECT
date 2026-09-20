# IDRA-CAPSTONE-PROJECT
Predicting e-commerce purchase intent from session behaviour — leakage-audited classification on 25k sessions (ROC-AUC 0.76)
# Predicting Online Purchase Intent from E-Commerce Session Behaviour

Analysing 25,000 e-commerce browsing sessions to identify what drives purchase completion, and building a classification model that predicts purchase intent using only signals available **before** the purchase decision.

Submitted as the Final Capstone Project for the Data Science & AI Summer Training Program, India Data Research Academy (IDRA) — Project Topic 10.

---

## Problem

Most browsing sessions on an e-commerce site never end in a purchase. Retailers can see a visitor's device, acquisition channel, pages viewed, time on site, and cart activity in real time — but have no reliable way to tell, mid-session, who is about to buy and who is only browsing.

**Question:** Can purchase completion be predicted from session-level behavioural, device, and marketing data available before the purchase decision is made?

---

## Dataset

`P_10_Ecommerce.csv` — 25,000 rows × 29 columns, one row per session, covering January–December 2024.

| | |
|---|---|
| Target | `purchased` (binary) |
| Positive class rate | 22.46% (5,616 sessions) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Outliers (IQR method) | 0 |

The dataset required no imputation or outlier removal. The categorical columns are supplied as numeric codes with no accompanying data dictionary; descriptive labels used in charts are documented as assumptions in the report.

---

## Approach

**Data leakage was the defining constraint of this project.**

Six columns — `rating`, `review_text`, `review_helpful_votes`, `cart_abandoned`, `revenue`, `revenue_normalized` — are only populated after (or as a direct consequence of) a purchase. `revenue` is exactly 0 for every non-purchasing session, and the review fields correlate with the target at r > 0.8. Including them would produce a model with near-perfect accuracy and zero real-world use, since none of these values exist at the moment a prediction would need to be made.

All six were excluded from the feature set. Every result below reflects that constraint.

**Pipeline:**

```
Data understanding
  → Cleaning (missing values, duplicates, outliers, leakage audit)
  → Exploratory data analysis
  → Statistical testing (chi-square, Welch's t-test)
  → Feature engineering (engagement_score, is_weekend)
  → Preprocessing (one-hot encoding, standardisation, stratified 80/20 split)
  → Modelling (Logistic Regression, Random Forest)
  → Evaluation
```

Both models were trained with `class_weight='balanced'` to address the 22/78 class imbalance — without it, the model simply predicts "not purchased" for nearly every session, scoring high accuracy with near-zero recall on the class that matters.

---

## Results

| Model | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 57.3% | 0.34 | 1.00 | 0.51 | 0.763 |
| **Random Forest** *(final)* | **61.0%** | **0.36** | **0.91** | **0.51** | **0.761** |

Random Forest was selected as the final model. Logistic Regression's near-perfect recall comes from classifying almost every session as a purchase, which makes it uninformative despite the comparable F1 and AUC; Random Forest achieves high recall while still separating the classes, and its feature importances are interpretable.

**Interpreting these numbers:** an ROC-AUC of 0.76 is a meaningful improvement over chance (0.50) but is not a high-confidence classifier. At 0.36 precision, roughly two of every three sessions flagged as likely purchases will not convert. This is a realistic ceiling once leakage is removed — pre-decision session signals only partially separate buyers from browsers.

The model is tuned toward recall on purpose: in a retention-targeting use case, missing a genuine buyer costs more than showing an unnecessary offer to a non-buyer.

---

## Key Findings

**1. Cart addition dominates.** `added_to_cart` accounts for 67% of total feature importance — an order of magnitude above any other variable.

| Feature | Importance |
|---|---|
| `added_to_cart` | 0.674 |
| `time_on_site_sec` | 0.038 |
| `engagement_score` | 0.036 |
| `unit_price` | 0.036 |
| `discount_amount` | 0.025 |

**2. Device type and marketing channel do not predict conversion.** Chi-square tests of independence returned p = 0.63 and p = 0.34 respectively — no significant association at the 5% level. Conversion rates across all six channels fall within a narrow 21.7–23.6% band. These dimensions appear to affect traffic volume, not conversion quality.

**3. Time on site is significant but modest.** Purchasing sessions averaged 929.7s versus 895.6s for non-purchasing sessions (Welch's t-test, p < 0.001). Statistically real, practically small.

**4. Pages viewed alone tells you nothing.** Median of 13 pages in both groups; t-test p = 0.33. Depth of browsing only becomes informative when combined with duration — the engineered `engagement_score` (`pages_viewed × time_on_site_sec`) ranked third in feature importance, above either of its source variables.

---

## Repository Contents

```
├── IDRA_Capstone_Report.pdf        Full report — problem definition through recommendations
├── IDRA_Capstone_Notebook.ipynb    Complete analysis: cleaning, EDA, statistics, modelling
├── P_10_Ecommerce.csv              Dataset
└── README.md
```

---

## Running the Analysis

1. Open `IDRA_Capstone_Notebook.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the data-loading cell —upload the P_10_Ecommerce file
3. `Runtime → Run all`.

The notebook is self-contained just upload the P_10_Ecommerce file in the setup and data loading part and it runs end to end without modification.

**Stack:** Python · pandas · NumPy · scikit-learn · SciPy · matplotlib · seaborn

---

## Limitations

- The categorical codes have no published data dictionary; category labels are reasonable assumptions, not confirmed mappings.
- At 0.36 precision, any intervention triggered by this model reaches a majority of non-buyers — acceptable for low-cost actions like an exit-intent prompt, not for high-cost ones.
- The dataset appears to be simulated rather than production traffic, so effect sizes should not be generalised to a live platform without revalidation.
- The `location` field (225 distinct codes, no dictionary) was excluded entirely.

---



Supervised by Dr. Shaheena Salam, India Data Research Academy
