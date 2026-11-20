# FraudShield – ML Handover Notes

## 1. Dataset Information

- Dataset: PaySim Synthetic Financial Transactions
- Total records: 6,362,620
- Total columns: 13
- Target column: `isFraud`

## 2. Dataset Location

`outputs/engineered_data/`

This is a Spark CSV output directory. Read the directory using PySpark rather than treating it as a single CSV file.

## 3. Target Distribution

| Class | Records |
|---|---:|
| Legitimate (0) | 6,354,407 |
| Fraud (1) | 8,213 |

The dataset is highly imbalanced. Accuracy alone is not sufficient for evaluating the model.

## 4. Engineered Features

| Feature | Description |
|---|---|
| sender_balance_change | oldbalanceOrg - newbalanceOrig |
| destination_balance_change | newbalanceDest - oldbalanceDest |
| log_amount | Natural logarithm of (1 + amount) |
| sender_balance_mismatch | 1 if sender balance calculation differs by more than 0.01; otherwise 0 |
| destination_balance_mismatch | 1 if destination balance calculation differs by more than 0.01; otherwise 0 |

## 5. Important Columns

- `isFraud`: Target variable.
- `type`: Categorical transaction type.
- `step`: Hourly simulation time-step, not actual clock hour.

## 6. Model Preparation Notes

- Exclude `isFraud` from input features because it is the target.
- Do not use `isFlaggedFraud`; it was excluded during preprocessing.
- Exclude raw account identifiers; they were removed during preprocessing.
- Encode `type` using Spark MLlib StringIndexer and OneHotEncoder.
- Consider class imbalance handling and appropriate evaluation metrics.
- Evaluate precision, recall, F1-score, PR-AUC and confusion matrix rather than relying only on accuracy.
- Compare performance with and without the balance mismatch features. Their usefulness is not guaranteed.

## 7. Validation Results

- Total records: 6,362,620
- Missing values: 0
- Fraud records: 8,213
- Legitimate records: 6,354,407
- All five engineered features verified.