# FraudShield - Machine Learning Methodology

This document explains *why* the ML side of FraudShield is built the way it is.
Every number quoted in the project README comes from the JSON and CSV files in
`outputs/`, which are produced by the scripts described below - none of them are
typed in by hand.

---

## 1. Where Member 2's work starts

Member 1 produced an engineered dataset of 6,362,620 PaySim transactions with
13 columns: the 8 preprocessed columns (`step`, `type`, `amount`,
`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`,
`isFraud`) plus 5 engineered features (`sender_balance_change`,
`destination_balance_change`, `log_amount`, `sender_balance_mismatch`,
`destination_balance_mismatch`).

Member 2's work begins by *verifying* that dataset rather than trusting the
handover notes (`src/verify_dataset.py`), then builds the ML pipeline on top of
it. No Member 1 file was modified.

One addition was made for performance: `src/convert_to_parquet.py` writes a
Parquet copy of the same data. CSV must be re-parsed as text on every read;
Parquet stores column types and compresses each column, so the repeated reads
that training, tuning and evaluation need become several times cheaper. The
script verifies that the copy has exactly the same row and fraud counts. It
also adds a `row_id` column, used only to identify a transaction in an alert -
never as a model input.

---

## 2. Feature decisions

### Included

| Feature | Why |
|---|---|
| `amount`, `log_amount` | Transaction size. The log version compresses a range that spans 0 to 10 million, which helps the linear model. |
| `oldbalanceOrg`, `newbalanceOrig` | The sender's balance before and after. |
| `oldbalanceDest`, `newbalanceDest` | The destination's balance before and after. |
| `sender_balance_change` | How much left the sender's account. |
| `destination_balance_change` | How much arrived at the destination. |
| `sender_balance_mismatch`, `destination_balance_mismatch` | Whether the balances moved by the amount they should have. Kept only after the ablation in §7 showed what they contribute. |
| `type` (one-hot encoded) | Transaction type. Fraud only ever occurs in TRANSFER and CASH_OUT, so this is a strong, legitimate signal. |

### Deliberately excluded

| Column | Why it is not a feature |
|---|---|
| `isFraud` | It is the target. |
| `isFlaggedFraud` | Already dropped by Member 1. It is the output of the bank's own rule engine, so using it would be **target leakage** - the model would be learning another detector's answer, not detecting fraud. |
| `nameOrig`, `nameDest` | Already dropped by Member 1. Account identifiers are unique strings; a model can memorise them but cannot generalise to accounts it has never seen. |
| `step` | **Used only to build the split, never as a feature.** `step` is simulation time. In PaySim the *proportion* of fraud rises in later steps because the simulator slows down legitimate activity while fraud continues at a steady rate. A model given `step` would learn "later means more fraud", which is an artefact of the generator and would be worthless on real data. |
| `row_id` | A display identifier added by us. Feeding a row number to a model is meaningless. |

### Leakage review

The engineered features are all computed from columns recorded *at the time of
the transaction*, so none of them contains information from the future or from
the label. The one feature family that needed thought is the mismatch pair: it
is derived from the same balance columns the model already sees, so at worst it
is redundant, not leaky.

---

## 3. Train / validation / test split (chronological)

PaySim has a time column, so a random split would be the wrong choice: it would
let the model train on transactions that happened *after* the ones it is tested
on, which cannot happen in a real deployment and inflates the scores.

The split follows `step`, with boundaries chosen after measuring the cumulative
row count so that the three windows are close to 70 % / 15 % / 15 % of the rows:

| Split | Steps | Share of rows |
|---|---|---|
| Train | 1 - 323 | ~70 % |
| Validation | 324 - 378 | ~15 % |
| Test | 379 - 743 | ~15 % |

Training rows are strictly earlier than validation rows, which are strictly
earlier than test rows (checked by `tests/test_integration.py`, check 5).

### The honest caveat

Fraud is a much larger *share* of the test window than of the training window,
because PaySim's legitimate transaction volume collapses after roughly step 400
while fraud continues at a steady rate. The exact figures are in
`outputs/model_results/dataset_verification.json`.

This matters when reading the results: precision and PR-AUC both depend on how
common the positive class is, so the test precision is flattered relative to
validation precision. Recall is not affected by prevalence. We kept the
chronological split anyway, because the alternative - a random split - would
hide a much more serious problem (training on the future) behind nicer-looking
numbers. The split boundaries are defined once in `src/config.py` so that every
script agrees.

---

## 4. Class imbalance

Fraud is about 0.1291 % of the data - roughly 1 transaction in 775.

**Why accuracy is useless here.** A model that answers "legitimate" every time
scores 99.87 % accuracy and catches zero fraud. Accuracy is dominated by the
6.35 million legitimate rows, so it cannot distinguish a useful model from a
useless one. The project reports fraud-class precision, recall, F1, F2 and
PR-AUC instead, and prints accuracy only to demonstrate the point.

**What we do about the imbalance: class weighting.** Each training row gets a
weight of `total_rows / (2 x rows_in_its_class)`, so the small fraud class and
the large legitimate class contribute equally to the loss. Spark MLlib's
`LogisticRegression` and `RandomForestClassifier` both accept a `weightCol`, so
this needs no change to the data itself.

**What we deliberately do not do.**

* *No SMOTE.* Synthetic minority oversampling would have to invent millions of
  fake fraudulent transactions. On this dataset it would also be applied in
  Spark with no mature implementation, and synthetic financial rows can easily
  be unrealistic (negative balances, impossible balance arithmetic).
* *No resampling before splitting.* Any resampling must happen strictly inside
  the training window; otherwise copies of the same transaction can land in both
  training and test, which leaks.
* *No undersampling of the majority class.* Throwing away 6 million legitimate
  transactions would discard exactly the information that teaches the model what
  normal behaviour looks like - and we have the compute to use them.

Class weighting has one visible side effect: it pushes predicted probabilities
upwards, which is why the default 0.5 cut-off produces far too many alerts and
why §6 exists.

---

## 5. The two models

Both are Spark MLlib models inside a single `Pipeline`, so the exact same
transformations are applied at training time and at scoring time.

```
type ──StringIndexer──▶ type_index ──OneHotEncoder──▶ type_ohe ──┐
                                                                 ├──▶ VectorAssembler ──▶ features ──▶ classifier
10 numeric columns ─────────────────────────────────────────────┘
                                   (+ StandardScaler, Logistic Regression only)
```

**Logistic Regression** (`src/train_logistic_regression.py`). A linear model:
it multiplies each feature by a weight and adds them up. Fast on millions of
rows and directly interpretable - a positive coefficient means "more of this
pushes towards fraud". Features are standardised first, because raw amounts
reach 10 million while `log_amount` stays under 17, and a regularised linear
model is sensitive to that. The regularisation strength `regParam` is chosen by
trying several values and keeping the best validation PR-AUC.

**Random Forest** (`src/train_random_forest.py`). Many decision trees, each
grown on a subsample of the rows and a random subset of features, voting
together. It captures *combinations* of conditions - "a TRANSFER, for an amount
equal to the sender's entire balance, where the destination balance barely
moves" - which a linear model cannot express. No scaling is applied, because a
tree compares one feature against a threshold at a time and scaling cannot
change which split it picks. The forest shape (`numTrees`, `maxDepth`) is
chosen on validation PR-AUC.

**Why not more models.** Gradient-boosted trees or a neural network would add
tuning burden and training time without changing the story this project tells;
the brief for this project fixes the scope at these two.

**Why `CrossValidator` is not used.** Spark's k-fold cross-validation shuffles
rows across folds, which would mix future transactions into training and defeat
the chronological split. A plain loop over hyperparameters, scored on the
chronological validation window, is the correct equivalent here.

**Reproducibility.** A fixed seed (`config.RANDOM_SEED = 42`) is used for the
forest's subsampling and for every sample taken anywhere in the project.

---

## 6. Threshold selection (`src/threshold_optimization.py`)

A classifier outputs a probability; turning that into "alert" or "no alert"
needs a cut-off, and 0.5 is only sensible when the classes are balanced and the
two kinds of error cost the same. Neither holds here.

So precision, recall, F1 and F2 are measured at about 100 candidate thresholds
on the **validation** split, in a single pass over the data (all thresholds are
evaluated as separate aggregate expressions in one Spark job). Four policies are
recorded:

| Policy | Meaning |
|---|---|
| `best_f1` | Highest F1 - treats a missed fraud and a false alarm as equally bad |
| `best_f2` | Highest F2 - weights recall twice as heavily as precision |
| `recall_at_least_0.80` | Of the thresholds reaching 80 % recall, the most precise one |
| `precision_at_least_0.50` | Of the thresholds reaching 50 % precision, the one catching the most fraud |

**FraudShield uses `best_f2`.** A missed fraudulent transfer is a direct
financial loss; a false alarm costs an analyst a few minutes. The two errors are
not equally expensive, so the operating point should favour recall. F2 does that
while still penalising a threshold that floods the team with alerts - which is
why it is preferred over "just maximise recall", a target that is trivially met
at a threshold of 0, where every transaction is an alert and nothing has been
learned.

The chosen threshold, the full sweep and the alternatives are saved to
`outputs/model_results/*_threshold_policy.json` and `*_threshold_sweep.csv`.

**Model selection also happens here, on validation only.** The model with the
higher validation PR-AUC is recorded in
`outputs/model_results/final_model_selection.json`. Only after that does
`src/model_comparison.py` touch the test split, which is why the test numbers
can be read as an honest estimate of unseen-data performance.

---

## 7. The balance-mismatch ablation (`src/feature_ablation.py`)

Member 1's handover notes asked for this explicitly, and it is a fair question:
`sender_balance_mismatch` is raised on 79.8 % of all rows and
`destination_balance_mismatch` on 65.8 %. Part of that is structural - the
formula assumes the sender's balance falls by the amount, which is not true for
CASH_IN - so the flags partly re-encode the transaction type.

The script trains the same model twice on the same split, with and without the
two flags, changing nothing else, and compares validation PR-AUC. The result is
in `outputs/model_results/feature_ablation.json`.

**What it found: keep them.** On the full dataset, dropping the two flags cost
Logistic Regression 43.85 % of its validation PR-AUC (0.4205 -> 0.2923) and
Random Forest 31.35 % (0.9999 -> 0.7612). This does not contradict Member 1's
EDA conclusion - that balance inconsistency *on its own* is a weak indicator
remains true, since the flags fire on 79.8 % and 65.8 % of all rows. What the
ablation shows is that the flags are valuable *in combination* with the other
features: they tell the model which transactions follow the expected balance
arithmetic, and that context changes the meaning of the balance columns. A
feature can be useless univariately and still carry real information
conditionally, which is a good distinction to be able to explain in a viva.

---

## 12. Sanity check: the deterministic rule baseline

The Random Forest reached F1 ~ 0.9995 on the test split. A score that high
normally means something leaked, so `src/rule_baseline.py` checks whether
PaySim's fraud is separable with no machine learning at all, using one rule:

```
type IN (TRANSFER, CASH_OUT) AND amount == oldbalanceOrg AND newbalanceOrig == 0
```

That is: *someone emptied this account in a single move.*

On the test split this rule alone achieves **precision 1.0000 and recall
0.9730** (0.9911 recall on validation), with no model involved. Full figures in
`outputs/model_results/rule_baseline.json`.

Three conclusions, all of which belong in the report rather than being quietly
omitted:

1. The Random Forest is rediscovering a pattern the simulator built in, not
   performing a subtle feat. Its near-perfect scores describe PaySim.
2. It explains the gap between the two models. The rule is an AND of three
   conditions: a tree expresses it in three splits, while Logistic Regression
   sums weighted features and cannot represent a conjunction at all. That is a
   structural limitation, not a tuning problem, and it is why the linear model
   plateaus around F1 0.75 on identical features.
3. It is **not** target leakage. The rule uses only fields recorded at
   transaction time - no `isFraud`, no `isFlaggedFraud`, no account identifiers.

---

## 8. Metrics, in plain words

With fraud as the positive class:

| Term | Meaning here |
|---|---|
| TP | A fraudulent transaction the model flagged - caught |
| FN | A fraudulent transaction the model missed - the expensive error |
| FP | A legitimate transaction the model flagged - a false alarm an analyst has to clear |
| TN | A legitimate transaction correctly left alone |
| Precision = TP/(TP+FP) | Of the alerts we raised, how many were real fraud |
| Recall = TP/(TP+FN) | Of all the fraud that happened, how much we caught |
| F1 | Harmonic mean of precision and recall; high only when both are |
| F2 | Same idea, with recall weighted twice as heavily |
| PR-AUC | Area under the precision-recall curve; the right threshold-free summary for a rare class, because it ignores the enormous number of true negatives |
| ROC-AUC | Secondary only. It divides false positives by the 6.35 million legitimate rows, so it looks generous on imbalanced data |

**The trade-off.** Lowering the threshold catches more fraud (recall up) but
raises more false alarms (precision down); raising it does the opposite. There
is no single correct answer - it depends on how many alerts a fraud team can
review per day. That is precisely why the threshold is a documented, measured
choice in this project rather than a default value, and why the dashboard shows
the whole precision/recall-versus-threshold curve instead of one number.

---

## 9. Fraud alerts (`src/fraud_alerts.py`, `src/model_prediction.py`)

Scoring loads one saved `PipelineModel` that contains the feature
transformations *and* the classifier, so there is no way for the serving code to
transform features differently from the training code.

Each scored transaction gets a fraud probability, a predicted class (using the
tuned threshold, not 0.5) and a risk category. The category boundaries are
derived from precision measured on the validation split:

| Category | Rule |
|---|---|
| High Risk | Probability at or above the lowest threshold that reached **90 % precision** on validation |
| Medium Risk | At or above the lowest threshold that reached **50 % precision** |
| Low Risk | At or above the operating threshold, but below Medium |

If a precision level turns out to be unreachable on validation, the band falls
back to the highest-precision threshold measured and the fallback is written
into the saved policy file, so it appears in the dashboard and the report rather
than being hidden.

`row_id` is carried through so an analyst can point at a specific transaction.
It is never a model input.

**This is batch detection, not a live system.** The script reads a file of
completed transactions, scores them, and writes a list of suspicious ones to
Parquet and CSV. It is not connected to a bank, a card network or a payment
switch, and it does not block or reverse anything. In a real deployment the same
saved pipeline would sit behind a streaming job or a service call - the scoring
logic would be unchanged; only the source of the transactions would differ.

---

## 10. Why Spark for this project

* **Volume.** 6.36 million transactions and 557 MB of engineered CSV. Spark reads
  the 16 part files in parallel across cores, and the same code would run
  unchanged on a cluster against a dataset a hundred times larger.
* **One engine for the whole job.** Ingestion, SQL aggregation, feature encoding,
  model training, scoring and persistence all happen in Spark. There is no
  hand-off to a second tool that could disagree about types or features.
* **MLlib trains on the full dataset, not a sample.** Both models are fitted on
  all 4.46 million training rows.
* **Parquet.** Columnar, compressed, typed persistence between stages - the
  standard storage format in a Spark workflow.
* **Spark SQL** is used for the analytical aggregations in
  `src/verify_dataset.py` and `src/prepare_dashboard_data.py`, which is the
  natural way to express group-by analytics over millions of rows.
* **The scaling consideration that shaped the design.** Streamlit re-runs its
  script on every click, so the dashboard must never touch the full dataset.
  Spark computes all analytics once and writes small aggregate files; the
  dashboard reads only those. Heavy computation distributed, serving layer thin.

Pandas appears only where it belongs: converting already-aggregated results
(tens to thousands of rows) into CSV files and rendering them in the dashboard.
It is never used for dataset-scale processing.

---

## 11. Hardware limits and what they changed

Development ran on a 2-core/8-thread Intel laptop with 8 GB of RAM, in Spark
local mode. This affected *settings*, not the method:

* The driver gets 5 GB and shuffle partitions are reduced from Spark's default
  200 to 16, because hundreds of tiny tasks waste more time on scheduling than
  they gain in parallelism on 8 cores.
* Cached feature vectors are persisted `MEMORY_AND_DISK`, so Spark spills to
  disk instead of failing when 4.46 million feature vectors do not fit in RAM.
* The Random Forest hyperparameter grid is small (two configurations), and each
  tree is grown on 60 % of the rows. On a cluster the same script would run a
  wider grid by changing one list.

Both final models were trained on the **complete** dataset. The `--sample` flag
exists for fast debugging, and any run that used it is marked
`"trained_on": "sample"` in its saved summary, so sample results can never be
mistaken for full-scale results.
