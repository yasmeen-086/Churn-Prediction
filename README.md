# Customer Churn Prediction

> Predict which telecom customers are likely to churn using Logistic Regression and Random Forest.

## Results
| Metric | Score |
|---|---|
| ✅ Accuracy | **87%** |
| ✅ ROC-AUC | **0.88** |

## Tech Stack
`Python` `Pandas` `Scikit-learn` `Matplotlib` `Seaborn`

## Dataset
- **7,043** customer records · **20+ features**
- Source: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Or use the built-in synthetic data generator (no download needed)

## Project Structure
```
churn_prediction/
├── churn_model.py                  
├── requirements.txt
└── README.md
```

## Quickstart
```bash
pip install -r requirements.txt

# Run the full pipeline
python churn_model.py

# Or open the notebook
jupyter notebook Customer_Churn_Prediction.ipynb
```

## Features Engineered
| Feature | Description |
|---|---|
| `tenure_ratio` | Normalised customer loyalty (0–1) |
| `contract_duration` | Numeric duration: 1 / 12 / 24 months |
| `avg_monthly_spend` | TotalCharges ÷ tenure |
| `num_services` | Count of subscribed add-ons |
| `is_senior_no_partner` | High-risk demographic flag |

## Key Findings
- Month-to-month contract customers churn at **3–4× the rate** of annual subscribers
- Short tenure (< 6 months) is the **strongest churn predictor**
- Fiber optic internet users churn more — likely due to **high monthly charges**
- Adding `Online Security` or `Tech Support` significantly **reduces churn risk**

## Models
Both models use a **scikit-learn Pipeline** (StandardScaler → Classifier) with:
- **Class weighting** to handle the ~26% churn imbalance
- **5-fold cross-validation** for robust AUC estimates
- **Feature importance** analysis on Random Forest
