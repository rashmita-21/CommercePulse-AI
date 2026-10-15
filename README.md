# CommercePulse AI — Retail Analytics & Repeat-Purchase Prediction

CommercePulse AI is an end-to-end retail analytics and machine learning project built using the **UCI Online Retail II dataset**. It combines data preparation, business intelligence, customer behaviour analysis, and predictive modelling to help businesses understand sales performance and identify customers who are likely to make repeat purchases.

The project connects historical retail transactions with actionable business insights and a machine learning pipeline designed to support customer retention decisions.

## Project Objectives

- Analyse historical retail sales, customer purchasing patterns, product performance, and country-level performance.
- Prepare and audit transaction data for reliable downstream analysis.
- Build a Power BI dashboard for business performance monitoring.
- Predict whether a customer will make a repeat purchase within the following three months.
- Compare machine learning models using a time-based evaluation strategy.
- Generate customer-level predictions and feature importance reports for further analysis.

## Tech Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Business intelligence | Power BI |
| Data storage | CSV files |
| Version control | Git, GitHub |

## Project Workflow

1. **Data inspection and auditing** — inspect the source data, identify data-quality issues, and assess transaction records.
2. **Data preparation** — clean and prepare transaction data for downstream analysis.
3. **Business analytics** — generate sales trends, product performance, country-level analysis, and customer behaviour reports.
4. **Customer feature engineering** — construct historical customer snapshots using information available at each observation date.
5. **Predictive modelling** — train Logistic Regression and Random Forest classifiers to predict repeat purchasing.
6. **Temporal evaluation** — evaluate models on a later historical period rather than relying solely on a random train-test split.
7. **Reporting** — save model metrics, classification results, feature importance, and customer-level predictions.

## Machine Learning: Repeat-Purchase Prediction

The predictive component estimates whether a customer will make a repeat purchase during the three months following an observation date.

### Modelling approach

- **Observation dates:** July 2010, October 2010, January 2011, and April 2011 for training snapshots.
- **Temporal test snapshot:** July 1, 2011.
- **Prediction horizon:** The following three months.
- **Training snapshots:** 15,595 customer-snapshot records.
- **Temporal test set:** 5,041 customers.
- **Repeat-purchase rate in the test set:** 34.99%.

Customer features include historical purchase recency, transaction frequency, purchase quantity, sales value, and product diversity.

### Model comparison

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| Accuracy | 76.31% | 75.42% |
| Precision | 65.27% | 63.40% |
| Recall | 69.05% | 70.41% |
| F1 Score | 67.11% | 66.72% |
| ROC AUC | 82.02% | 81.68% |
| Average Precision | 74.08% | 73.02% |

**Selected model: Random Forest**, based on its higher ROC AUC in this evaluation.

Random Forest performed slightly better overall on the reported ROC AUC, F1 score, and average precision metrics. Logistic Regression achieved slightly higher recall, which may be relevant when the business objective is to identify as many potential repeat purchasers as possible.

These results represent a historical backtest on the available dataset. They do not guarantee performance on present-day customers or establish the financial impact of a retention campaign.

## Business Applications

CommercePulse AI demonstrates how retail transaction data can support:

- **Customer retention:** Identify customers with a higher predicted likelihood of returning.
- **Campaign prioritisation:** Help marketing teams prioritise customers for retention experiments.
- **Sales monitoring:** Track monthly sales and identify changes in performance.
- **Product decisions:** Understand which products contribute most to sales.
- **Market analysis:** Compare sales performance across countries.
- **Data-driven planning:** Combine descriptive analytics with predictive modelling.

Predictions should be validated and paired with campaign experiments before being used to allocate actual marketing budgets.

## Repository Structure

```text
CommercePulse-AI/
├── data/
│   ├── raw/                 # Source dataset (not committed)
│   └── processed/           # Prepared data (not committed)
├── models/                  # Saved model artefacts (not committed)
├── reports/
│   ├── ml_classification_report.txt
│   ├── ml_feature_importance.csv
│   ├── ml_model_comparison.csv
│   ├── ml_model_metrics.csv
│   └── processing_summary.csv
├── src/
│   ├── inspect_data.py
│   ├── audit_data.py
│   ├── prepare_data.py
│   ├── business_insights.py
│   └── predict_repeat_purchase.py
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rashmita-21/CommercePulse-AI.git
cd CommercePulse-AI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

```bash
python -m pip install pandas numpy scikit-learn openpyxl joblib
```

### 4. Obtain the dataset

Download the **Online Retail II** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

Place the Excel file at:

```text
data/raw/online_retail_II.xlsx
```

The raw dataset and processed transaction files are excluded from this repository.

### 5. Run the pipeline

Execute the scripts from the project root, following the data dependencies:

```bash
python src/inspect_data.py
python src/audit_data.py
python src/prepare_data.py
python src/business_insights.py
python src/predict_repeat_purchase.py
```

The scripts generate processed data, analytical reports, and machine learning outputs locally. The exact outputs depend on the dataset and execution environment.

## Power BI Dashboard

The project also includes a Power BI dashboard developed for retail performance monitoring, with:

- Key performance indicator cards
- Monthly sales trends
- Top 10 products
- Country-level sales performance
- Date filtering

The existing dashboard is maintained separately from the Python modelling pipeline.


## Author

**Rashmita Perumanla**

GitHub: [rashmita-21](https://github.com/rashmita-21)

---

*CommercePulse AI demonstrates an integrated approach to retail analytics, customer behaviour analysis, and predictive modelling using historical transaction data.*