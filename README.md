# FraudShield - Financial Fraud Detection Using Big Data Analytics

A fraud detection system for financial transactions built on **Apache Spark**
(PySpark 4.3.0) and **Spark MLlib**, with an interactive **Streamlit**
dashboard. The project processes the full 6,362,620-row PaySim transaction
dataset - ingestion, analytics, feature engineering, model training, threshold
tuning, alert generation and visualisation all run through Spark.

B.Tech mini project, two members. Member 1 built the data engineering and
analytics layer; Member 2 built the machine learning, evaluation, fraud alerting
and dashboard layer.

---

## 1. Problem statement

Financial fraud is rare but expensive. In this dataset only **0.1291 %** of
transactions are fraudulent - about 1 in 775 - yet they account for over
12 billion in transaction value. Two things make the problem hard:

1. **Scale.** 6.36 million transactions do not fit comfortably in a single
   machine's memory with pandas, and a real bank processes far more. Processing
   has to be distributed.
2. **Extreme class imbalance.** A model that labels everything "legitimate"
   achieves 99.87 % accuracy and catches zero fraud. Accuracy is therefore
   actively misleading, and the system must be measured on how much fraud it
   catches and how many false alarms it raises.

A fraud detection system must also decide *where to draw the line*: flag too
little and fraud goes through, flag too much and the investigation team drowns
in false alarms.

## 2. Objectives

1. Ingest and process a large financial transaction dataset with Apache Spark.
2. Analyse transaction behaviour and identify suspicious patterns.
3. Engineer features that describe how money and balances move.
4. Train and compare two machine learning models using Spark MLlib.
5. Evaluate them with metrics appropriate to a rare positive class.
6. Choose a fraud decision threshold from measured evidence, not a default.
7. Generate fraud alerts with risk categories.
8. Present everything through an interactive dashboard built on real outputs.

## 3. Dataset

**PaySim** - a synthetic dataset that simulates mobile money transactions,
generated from real transaction logs of an African mobile money service. It is
used because real financial transaction data cannot be published.

| Property | Value (measured, not quoted) |
|---|---|
| Total transactions | 6,362,620 |
| Fraudulent | 8,213 (0.1291 %) |
| Legitimate | 6,354,407 (99.8709 %) |
| Columns after engineering | 13 |
| Missing values | 0 |
| Time span | steps 1-743 (one step = one simulated hour, ~31 simulated days) |
| Transaction types | CASH_OUT 2,237,500 · PAYMENT 2,151,495 · CASH_IN 1,399,284 · TRANSFER 532,909 · DEBIT 41,432 |
| Types containing fraud | TRANSFER (4,097) and CASH_OUT (4,116) only |
| Engineered dataset size | 557 MB (16 Spark CSV part files) |

`step` is a simulated hourly time step, **not** a real clock hour.

Two findings that shape the modelling:

* **98.05 % of fraudulent transactions leave the sender's balance at exactly
  zero.** The account is drained. This is the single strongest signal in the
  data.
* **Fraud occurs only in TRANSFER and CASH_OUT.** PaySim's fraud agents take
  over an account, transfer the balance out, and cash it out.

### Dataset location

The dataset is not in this repository (it is far too large for git). The project
reads it from `outputs/engineered_data`, which can be a symlink to wherever the
folder actually lives, or the path can be set with the
`FRAUDSHIELD_ENGINEERED_DATA` environment variable. See
[docs/SETUP.md](docs/SETUP.md) §4.

## 4. Big Data technologies used

| Layer | Technology |
|---|---|
| Language | Python 3.11.18 |
| Distributed processing | Apache PySpark 4.3.0 (`local[*]`, 8 cores) |
| JVM | OpenJDK 17 (Temurin) - required by Spark |
| Machine learning | Spark MLlib (`Pipeline`, `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `BinaryClassificationEvaluator`) |
| Analytical queries | Spark SQL |
| Storage | Spark DataFrames, Parquet (snappy), CSV |
| Dashboard | Streamlit 1.42 + Plotly 6.0 |
| Small-scale serving | pandas (aggregated summaries only) |

**Why Spark and not pandas + scikit-learn?** The dataset is 557 MB engineered
(and the raw PaySim file is larger); a bank's would be terabytes. Spark reads
the 16 part files in parallel, distributes the feature encoding and the model
fitting across cores, spills to disk rather than failing when memory runs out,
and runs the *same code* on a cluster. Every model here is trained on all
4.46 million training rows, not a sample. pandas is used only to turn
already-aggregated results (tens to thousands of rows) into dashboard files.

## 5. System architecture

```
                 ┌──────────────────── MEMBER 1 ────────────────────┐
dataset/paysim.csv
        │
        ▼
  data_ingestion.py ──▶ data_quality.py, 19 EDA scripts (Spark SQL + DataFrames)
        │
        ▼
  data_preprocessing.py   drop nameOrig/nameDest/isFlaggedFraud, keep 8 columns
        │                 ──▶ outputs/cleaned_data/
        ▼
  feature_engineering.py  + 5 engineered features
        │                 ──▶ outputs/engineered_data/   (13 columns, 6.36 M rows)
        ▼
  validate_engineered_data.py
                 └──────────────────────────────────────────────────┘
        │
                 ┌──────────────────── MEMBER 2 ────────────────────┐
        ▼
  convert_to_parquet.py ──▶ outputs/engineered_parquet/  (+ row_id for display)
        │
        ▼
  verify_dataset.py ──────▶ outputs/model_results/dataset_verification.json
        │
        ▼
  ml_preprocessing.py     chronological split, class weights, MLlib pipeline
        │
        ├──▶ train_logistic_regression.py ──▶ models/logistic_regression/
        └──▶ train_random_forest.py ───────▶ models/random_forest/
        │
        ▼
  threshold_optimization.py   sweep ~100 thresholds on VALIDATION
        │                     ──▶ *_threshold_policy.json, *_threshold_sweep.csv
        │                     ──▶ final_model_selection.json
        ▼
  model_comparison.py     TEST split evaluated ONCE, after the threshold is fixed
        │                 ──▶ model_comparison.{json,csv}, confusion_matrices.json
        ▼
  model_prediction.py ──▶ fraud_alerts.py ──▶ outputs/fraud_alerts/
        │
        ▼
  prepare_dashboard_data.py ──▶ outputs/dashboard_data/  (small aggregates)
        │
        ▼
  dashboard/app.py  (Streamlit; reads only the precomputed files)
                 └──────────────────────────────────────────────────┘
```

## 6. Data engineering workflow (Member 1)

1. **Ingestion** - PaySim CSV loaded into a Spark DataFrame (`local[*]`).
2. **Data quality** - missing values, duplicates, negative amounts, categorical
   validity and numeric ranges checked.
3. **Exploratory analysis** - 19 Spark scripts covering fraud distribution by
   type, fraud percentage by type, amount distributions, average/maximum fraud
   amounts, gross fraud value, fraud amount by type, time analysis over `step`,
   sender and destination balance behaviour, and balance-mismatch analysis.
4. **Preprocessing** (`src/data_preprocessing.py`) - keeps `step`, `type`,
   `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`,
   `newbalanceDest`, `isFraud`; drops `nameOrig`, `nameDest` (unique account
   identifiers, no predictive value) and `isFlaggedFraud` (output of the bank's
   own rule engine - using it would be target leakage).
5. **Feature engineering** (`src/feature_engineering.py`) - the five features
   below.
6. **Validation** (`src/validate_engineered_data.py`).

### Engineered features

| Feature | Definition |
|---|---|
| `sender_balance_change` | `oldbalanceOrg - newbalanceOrig` - how much left the sender |
| `destination_balance_change` | `newbalanceDest - oldbalanceDest` - how much arrived |
| `log_amount` | `ln(1 + amount)` - compresses a 0 to 10,000,000 range |
| `sender_balance_mismatch` | 1 if `|oldbalanceOrg - amount - newbalanceOrig| > 0.01` |
| `destination_balance_mismatch` | 1 if `|oldbalanceDest + amount - newbalanceDest| > 0.01` |

Member 2 re-derived all five from the raw columns across all 6.36 million rows:
**zero rows disagreed** with these formulas
(`outputs/model_results/dataset_verification.json`).

Member 1's own analytical conclusion on the mismatch features, preserved in
[docs/MEMBER1_EDA_FINDING.md](docs/MEMBER1_EDA_FINDING.md), was that balance
inconsistency alone is not an effective fraud indicator. Member 2 tested that
claim directly with a feature ablation (§10).

## 7. Machine learning methodology (Member 2)

Full reasoning in **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**. In brief:

**Features.** Ten numeric columns plus one-hot encoded `type`.
`isFraud` is the target; `isFlaggedFraud`, `nameOrig`, `nameDest` are excluded
as leakage or noise; **`step` is excluded from the features** and used only to
build the split, because in PaySim the fraud *share* rises in later steps as the
simulator slows legitimate activity - a model given `step` would learn that
artefact. `row_id` is a display identifier only.

**Chronological split**, not random - training on transactions that happened
after the test ones is impossible in production and inflates scores:

| Split | Steps | Rows | Fraud | Fraud rate |
|---|---|---|---|---|
| Train | 1-323 | 4,463,587 (70.15 %) | 3,643 | 0.0816 % |
| Validation | 324-378 | 980,416 (15.41 %) | 564 | 0.0575 % |
| Test | 379-743 | 918,617 (14.44 %) | 4,006 | 0.4361 % |

*Caveat stated openly:* the test window's fraud rate is ~5x the training
window's, because PaySim's legitimate volume collapses after about step 400
while fraud continues steadily. Precision and PR-AUC depend on class prevalence,
so test precision is flattered relative to validation. We kept the
chronological split because the alternative - a random split - hides a worse
problem (training on the future) behind nicer numbers.

**Class imbalance: weighting, not resampling.** Each training row is weighted
`total / (2 x class_count)`, so fraud (weight 612.6) and legitimate
(weight 0.5004) contribute equally to the loss. Both MLlib models accept a
`weightCol`. No SMOTE (it would fabricate millions of synthetic financial rows),
no undersampling (it would throw away the data that defines "normal"), and
nothing is ever resampled before the split.

**Models.** Logistic Regression (scaled features, L2 regularisation tuned on
validation) and Random Forest (no scaling - trees are scale-invariant; forest
shape tuned on validation). Spark's `CrossValidator` is deliberately *not* used:
k-fold shuffling would mix future rows into training.

**Threshold.** Not 0.5. ~100 thresholds are measured on validation in a single
Spark pass, and the operating point is chosen by the **F2** policy (recall
weighted twice as heavily as precision, because a missed fraudulent transfer
costs more than a false alarm an analyst clears in minutes).

**Model selection happens on validation only.** The test split is read exactly
once, by `src/model_comparison.py`, after the threshold is fixed.

<!-- RESULTS_SECTION_START -->
## 8. Results

All figures below were measured by running this repository on the PaySim engineered dataset. Both models were trained on the complete 4,463,587-row training window and evaluated on the held-out test window of 918,617 transactions containing 4,006 fraudulent ones (0.4361 %).

### Model comparison (test split, fraud = positive class)

| Model | Threshold | Precision | Recall | F1 | F2 | PR-AUC | ROC-AUC | TP | FP | FN | Alerts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.980 | 0.8605 | 0.6685 | 0.7525 | 0.6997 | 0.8195 | 0.9939 | 2,678 | 434 | 1,328 | 3,112 |
| RandomForestClassifier **(final)** | 0.670 | 0.9995 | 0.9995 | 0.9995 | 0.9995 | 1.0000 | 1.0000 | 4,004 | 2 | 2 | 4,006 |

### Reading these numbers

* **LogisticRegression** at threshold 0.980 caught 2,678 of 4,006 fraudulent transactions (66.85 % recall) and raised 434 false alarms, so 86.05 % of its alerts were genuine fraud. Its accuracy was 0.998082 - which says nothing useful, since always answering "legitimate" would have scored about 0.9956 on this split.
* **RandomForestClassifier** at threshold 0.670 caught 4,004 of 4,006 fraudulent transactions (99.95 % recall) and raised 2 false alarms, so 99.95 % of its alerts were genuine fraud. Its accuracy was 0.999996 - which says nothing useful, since always answering "legitimate" would have scored about 0.9956 on this split.

**Final model: RandomForestClassifier**, selected on highest validation PR-AUC (test split not used).

### What threshold tuning bought

| Model | | Threshold | Precision | Recall | F1 | Alerts raised |
|---|---|---:|---:|---:|---:|---:|
| LogisticRegression | default 0.5 | 0.500 | 0.0964 | 0.9930 | 0.1757 | 41,279 |
| LogisticRegression | **tuned** | 0.980 | 0.8605 | 0.6685 | 0.7525 | 3,112 |
| RandomForestClassifier | default 0.5 | 0.500 | 0.9970 | 0.9995 | 0.9983 | 4,016 |
| RandomForestClassifier | **tuned** | 0.670 | 0.9995 | 0.9995 | 0.9995 | 4,006 |

Class weighting pushes predicted probabilities upwards, so the default 0.5 cut-off floods the system with alerts. Tuning the threshold on the validation split is what turns a technically-trained model into a usable one.

### Validation versus test

| Model | Split | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| LogisticRegression | validation | 0.4349 | 0.6578 | 0.5236 | 0.4205 |
| LogisticRegression | test | 0.8605 | 0.6685 | 0.7525 | 0.8195 |
| RandomForestClassifier | validation | 0.9947 | 1.0000 | 0.9973 | 0.9998 |
| RandomForestClassifier | test | 0.9995 | 0.9995 | 0.9995 | 1.0000 |

Test precision and PR-AUC are higher than validation for both models, and that is mostly prevalence, not improvement: fraud is 0.4361 % of the test window against 0.0575 % of the validation window, and both metrics depend on how common the positive class is. Recall is prevalence-independent and is the fairer cross-split comparison.

### Chosen thresholds and risk bands

| Model | Policy | Operating threshold | High Risk >= | Medium Risk >= |
|---|---|---:|---:|---:|
| LogisticRegression | best_f2 | 0.980 | 0.999000 | 0.990000 |
| RandomForestClassifier | best_f2 | 0.670 | 0.999964 | 0.999954 |

* RandomForestClassifier: The model already exceeds 90 % precision at its operating threshold, so precision-based bands all collapsed onto that one value and every alert would have been High Risk. The bands are therefore taken from the 85th and 50th percentiles of the 567 flagged validation scores instead.

### Fraud alerts

`src/fraud_alerts.py` scored 918,617 transactions from the test split with RandomForestClassifier and raised **4,006 alerts** (0.4361 % of transactions), covering 6,322,314,930.80 in transaction value.

| Risk category | Alerts | Total amount | Average fraud probability |
|---|---:|---:|---:|
| High Risk | 489 | 55,803,288.61 | 1.0000 |
| Medium Risk | 1,623 | 4,424,311,323.16 | 1.0000 |
| Low Risk | 1,894 | 1,842,200,319.03 | 0.9918 |

Checked against the known labels in that data: 4,004 fraudulent transactions caught, 2 missed (recall 99.95 %), 2 false alarms (precision 99.95 %).

## 9. Feature ablation: do the balance-mismatch features help?

Member 1's handover notes asked for this check. The same model was trained twice on the same chronological split (the full dataset), changing only whether `sender_balance_mismatch` and `destination_balance_mismatch` were included.

| Model | PR-AUC with flags | PR-AUC without | Change | Verdict |
|---|---:|---:|---:|---|
| LR | 0.420510 | 0.292323 | +0.128187 (+43.85 %) | keep the mismatch features |
| RF | 0.999861 | 0.761223 | +0.238638 (+31.35 %) | keep the mismatch features |

## 10. Sanity check: is the dataset simply near-deterministic?

A test F1 of 0.9995 is the kind of number that usually means something has leaked, so before reporting it we checked whether PaySim's fraud is separable *without any machine learning*. `src/rule_baseline.py` evaluates one hand-written rule:

```
type IN (TRANSFER, CASH_OUT) AND abs(amount - oldbalanceOrg) < 0.01 AND newbalanceOrig = 0
```

In plain words: *someone emptied this account in a single move.*

| Split | Rows | Fraud | Rule matched | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 980,416 | 564 | 559 | 1.0000 | 0.9911 | 0.9955 |
| test | 918,617 | 4,006 | 3,898 | 1.0000 | 0.9730 | 0.9863 |

The rule alone reaches **precision 1.0000 and recall 0.9730** on the test split (0.9911 recall on validation), flagging 3,898 transactions of which 3,898 are fraud and 0 are not.

**Three conclusions follow, and all three belong in the report:**

1. *The Random Forest is not doing anything miraculous.* PaySim's fraud agents drain the victim's balance exactly, leaving an almost perfect signature in the balance columns. The forest rediscovers that conjunction; it does not transcend the data.
2. *This explains the large gap between the two models.* The rule is an AND of three conditions. A decision tree expresses it in three splits; Logistic Regression sums weighted features and cannot represent a conjunction at all, which is exactly why it plateaus around F1 0.75 on identical features.
3. *It is not target leakage.* The rule uses only fields recorded at transaction time - no `isFraud`, no `isFlaggedFraud`, no account identifiers. The separability is a property of the simulator, which is why the near-perfect scores must be read as describing PaySim and not as a claim about detecting real fraud.

<!-- RESULTS_SECTION_END -->

## 11. Fraud alert generation

`src/fraud_alerts.py` loads the saved pipeline, scores transactions, applies the
validation-chosen threshold and bands each alert:

| Risk category | Rule |
|---|---|
| High Risk | probability >= lowest validation threshold reaching 90 % precision |
| Medium Risk | probability >= lowest validation threshold reaching 50 % precision |
| Low Risk | probability >= the operating threshold, below Medium |

Every alert carries `row_id`, `step`, `type`, `amount`, the balance columns, the
predicted fraud probability, the predicted class and the risk category. Output
goes to `outputs/fraud_alerts/` as Parquet (complete) and CSV (for the dashboard
and downloads).

**This is a batch fraud detection demonstration.** It scores a file of completed
transactions. It is not connected to a bank, a card network or a live payment
feed, and it does not block or reverse anything. In production the same saved
pipeline would sit behind a streaming job or a service call - the scoring logic
would be identical; only the input source would change.

## 12. Dashboard

```bash
source env.sh && streamlit run dashboard/app.py
```

Five sections:

* **Overview** - total/fraud/legitimate counts, fraud percentage, fraud-labelled
  value, class balance, the split table, and the dataset verification results.
* **Transaction analytics** - volume and fraud rate by type, amount bands, amount
  statistics, fraud over simulation time (with a step-range filter), fraud by
  hour of simulated day, and balance behaviour.
* **Model performance** - the comparison table, grouped metric chart, confusion
  matrices, the precision/recall/F1-versus-threshold curve with the chosen
  threshold marked, candidate threshold policies, Random Forest feature
  importances and Logistic Regression coefficients.
* **Fraud detection** - the alert table with filters for transaction type, risk
  category, amount range and fraud probability, plus the risk breakdown.
* **Downloads** - every result file as CSV/JSON.

**Performance design.** Streamlit re-runs its script on every interaction, so
the dashboard never touches the 6.36 M-row dataset. `prepare_dashboard_data.py`
computes all analytics once in Spark and writes small aggregate files; the
dashboard reads only those (cached, keyed on file modification time) and uses a
stratified sample for interactive browsing. Heavy computation stays
distributed; the serving layer stays thin.

Every number shown is read from the files in `outputs/` - nothing is hard-coded,
and a missing file produces a message naming the script to run.

## 13. Installation

Full guide with troubleshooting: **[docs/SETUP.md](docs/SETUP.md)**.

```bash
java -version
```

Java 17 is mandatory (Spark runs on the JVM). On macOS: `brew install --cask temurin@17`,
or unpack the Adoptium archive into `~/Library/Java/JavaVirtualMachines` - see
[docs/SETUP.md](docs/SETUP.md) §2. On Linux: `sudo apt install openjdk-17-jdk`.

```bash
python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

```bash
ln -sfn /absolute/path/to/engineered_data outputs/engineered_data
```

```bash
source env.sh && python src/test_spark.py
```

## 14. Execution

```bash
./run_pipeline.sh
```

Runs every stage in order: Parquet conversion, dataset verification, both model
trainings, threshold optimisation, test comparison, fraud alerts, dashboard
aggregates and the integration tests. Add `--sample` for a fast 2 % debugging
run (clearly marked as sample results and never used for the report).

Individual stages, flags and environment variables are listed in
[docs/SETUP.md](docs/SETUP.md) §6-7.

```bash
python tests/test_integration.py
```

```bash
source env.sh && streamlit run dashboard/app.py
```

## 15. Project structure

```
FraudShield/
├── README.md                     this file
├── README_ML_HANDOVER.md         Member 1 -> Member 2 handover notes
├── requirements.txt
├── env.sh                        sets JAVA_HOME, venv, PYSPARK_PYTHON
├── run_pipeline.sh               runs the whole Member 2 pipeline
├── docs/
│   ├── METHODOLOGY.md            why the ML is built this way
│   ├── SETUP.md                  install, run, troubleshoot
│   ├── RESULTS.md                measured results in full
│   └── MEMBER1_EDA_FINDING.md    Member 1's original README text
├── src/
│   ├── (24 Member 1 scripts - unmodified)
│   ├── config.py                 paths, schema, split boundaries, Spark session
│   ├── convert_to_parquet.py
│   ├── verify_dataset.py
│   ├── ml_preprocessing.py       pipeline, class weights, splits
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── model_evaluation.py       metrics, confusion matrix, PR-AUC
│   ├── threshold_optimization.py
│   ├── model_comparison.py
│   ├── feature_ablation.py
│   ├── model_prediction.py
│   ├── fraud_alerts.py
│   └── prepare_dashboard_data.py
├── dashboard/
│   ├── app.py                    Streamlit application
│   └── theme.py                  palette and chart styling
├── models/                       saved Spark MLlib pipelines (git-ignored)
├── outputs/                      data, results, alerts (git-ignored)
└── tests/
    └── test_integration.py       16 end-to-end checks
```

Member 1's 24 scripts in `src/` were **not modified, renamed or deleted**.

<!-- LIMITATIONS_SECTION_START -->
## 16. Limitations

* **The data is synthetic.** PaySim is a simulation. Its fraud pattern is consistent (drain the account, transfer, cash out), which makes it more learnable than real fraud. These results should not be read as a prediction of performance on a real bank's traffic.
* **Prevalence shift across time.** The test window has a much higher fraud rate than the training window because of how PaySim tapers legitimate volume. Precision and PR-AUC are inflated on test relative to validation for that reason alone.
* **Batch, not streaming.** Alerts are generated by scoring a file. There is no live transaction feed, no blocking and no case management.
* **No account history features.** Each transaction is scored on its own fields. Real systems use per-account velocity and behavioural history, which PaySim's anonymised account names do not support well.
* **Single machine.** Everything ran in Spark local mode on an 8 GB laptop, which constrained the hyperparameter grid and forced cached feature vectors to spill to disk. The code is cluster-ready; the search was not exhaustive.
* **No model monitoring.** A deployed model would need drift detection and periodic retraining; neither is implemented.

## 17. Future enhancements

* Spark Structured Streaming for near-real-time scoring.
* Per-account velocity features (transactions and amounts in a rolling window).
* Gradient-boosted trees as a third model, with calibration curves.
* Cost-based threshold selection using an explicit cost per missed fraud and per false alarm.
* Alert triage workflow: analyst feedback captured and fed into retraining.
* Model explainability per alert (feature contributions shown next to each flagged transaction).

<!-- LIMITATIONS_SECTION_END -->

## 18. Member contributions

**Member 1 - Data engineering and analytics**
Environment setup; Spark ingestion of PaySim; data quality checks (missing
values, duplicates, negative values, categorical and numerical validity);
19 exploratory analysis scripts (fraud by type, amount distributions, fraud
amounts, time analysis, sender/destination balance analysis, balance mismatch
analysis); preprocessing and column selection; the five engineered features;
engineered-dataset validation.

**Member 2 - Machine learning, evaluation, alerts and dashboard**
Independent verification of the engineered dataset in Spark; Parquet conversion;
MLlib feature pipeline; chronological splitting; class-weight imbalance
handling; Logistic Regression and Random Forest training with validation-based
hyperparameter selection; the evaluation module (fraud-class precision, recall,
F1, F2, PR-AUC, confusion matrices); threshold optimisation and the documented
F2 operating-point policy; the balance-mismatch feature ablation; final test
comparison and model selection; fraud alert generation with validation-derived
risk bands; the Streamlit dashboard; the 16-check integration suite; project
documentation.

## 19. Academic honesty notes

* Every number in this README and in `docs/RESULTS.md` was produced by running
  the code in this repository on the actual PaySim engineered dataset. Nothing
  is estimated or illustrative.
* Both final models were trained on the **complete** 4.46 M-row training split.
  The `--sample` flag exists for debugging and any run that used it is marked
  `"trained_on": "sample"` in its saved summary.
* The test split was read once, after the thresholds and the final model were
  fixed on validation.
* The dataset is synthetic (PaySim). Results demonstrate the method; they are
  not claims about any real bank's fraud rate.
