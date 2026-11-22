# FraudShield — Demonstration Script and Viva Preparation

Everything here was rehearsed against the working system. Timings assume a
15-minute slot: 10 minutes of presentation, 3 minutes of live demo, 2 minutes
of questions. Adjust by cutting slides 13 and 16 first.

---

## 1. Before you walk in

Run this once, about ten minutes before the session, and leave it running:

```bash
cd /Users/sairamav/Downloads/FraudShield && source env.sh && python tests/test_integration.py
```

Expect `16 passed, 0 failed`. If anything fails, the message names the script
to re-run — do not start the demo until it is green.

Then start the dashboard and leave the browser open on the Overview page:

```bash
cd /Users/sairamav/Downloads/FraudShield && source env.sh && streamlit run dashboard/app.py
```

It serves at <http://localhost:8501>.

**Checklist**

- [ ] Integration tests green
- [ ] Dashboard open at Overview, browser zoomed so the KPI cards fill the screen
- [ ] A terminal open in the project directory with `env.sh` already sourced
- [ ] `docs/FraudShield_Presentation.pptx` open in presenter view (speaker notes are on every slide)
- [ ] Laptop on mains power — Spark will spin the fans
- [ ] Phone hotspot ready if the venue Wi-Fi is needed (nothing in the demo requires internet)

---

## 2. Presentation flow (10 minutes)

| Slide | Time | The one thing to land |
|---|---|---|
| 1 Title | 20 s | What it is: fraud detection on 6.36 M transactions with Spark |
| 2 The problem | 1 min | **1 in 775.** A model that says "legitimate" every time is 99.87 % accurate and useless |
| 3 Objectives | 30 s | Don't read all six — point at "evaluate honestly" and "choose a threshold" |
| 4 Dataset | 1 min | 98.05 % of frauds leave the sender's balance at exactly zero |
| 5 Architecture | 1 min | Left to right; one clean handover artefact between the two members |
| 6 Why Spark | 1 min | Heavy computation distributed, serving layer thin |
| 7 Features | 1 min | The *exclusions* matter most — leakage, and why `step` is not a feature |
| 8 Split + imbalance | 1 min | Chronological, not random; weighting, not SMOTE |
| 9 Why accuracy fails | 45 s | Say the 99.87 % number out loud again |
| 10 Threshold | 1 min | 41,279 alerts → 3,112, precision 9.6 % → 86.1 % |
| 11 Results | 1 min | Random Forest 99.95 % recall, 2 false alarms; chosen on validation |
| 12 Honesty check | 1.5 min | **The strongest slide.** A rule with no ML gets precision 1.0000 |
| 13 Ablation | 45 s | Cut this first if short on time |
| 14 Alerts | 45 s | Batch demonstration, not a live bank |
| 15 Dashboard | 30 s | Then switch to the live app |
| 16 Testing | 45 s | Three real bugs found — cut second if short on time |
| 17 Conclusion | 45 s | Three takeaways, then limitations |
| 18 Thank you | 10 s | Invite questions |

---

## 3. Live demo (3 minutes)

Switch to the browser. Do **not** improvise — follow this path.

**Overview (30 s).**
"These five numbers come from the pipeline's own output files, not from
anything typed into the dashboard. 6,362,620 transactions, 8,213 fraudulent.
Underneath is the chronological split — training ends at step 323, validation
runs to 378, test is everything after."

**Transaction analytics (40 s).**
Click *Transaction Analytics*. Point at the fraud-rate chart.
"Fraud only ever appears in TRANSFER and CASH_OUT. PAYMENT, CASH_IN and DEBIT
contain none at all."
Then drag the step-range slider.
"Legitimate volume collapses after about step 400 while fraud continues — that
is why the test window has a higher fraud rate than training, and why we are
careful about comparing precision across splits."

**Model performance (50 s).**
Click *Model Performance*. Point at the comparison table, then scroll to the
threshold curve.
"Precision climbs and recall falls as the threshold rises. The dashed line is
the point we chose on validation data. The red line at 0.5 is the default we
did *not* use — at 0.5 this model raises 41,279 alerts."

**Fraud detection (50 s).**
Click *Fraud Detection*. Point at the four KPI cards, then the risk bars.
"4,006 alerts, 4,004 of them genuine fraud, two false alarms."
Now use a filter — set the risk category to *High Risk* only.
"The table is filterable by type, risk, amount and probability, and an analyst
can export exactly what they filtered."

**Downloads (10 s).** Click *Downloads*, scroll once.
"Every result file is downloadable — this is the evidence behind the report."

---

## 4. If something breaks

| Symptom | What to do, calmly |
|---|---|
| Dashboard shows "not generated yet" | That file was deleted. Say "the dashboard tells you exactly which script regenerates it" — that *is* the designed behaviour. Move to the slides. |
| Dashboard won't start | Fall back to the deck: slide 15 has a full screenshot. Do not debug live. |
| Page is slow on first load | It is reading ~3.5 MB of aggregates. Say so; it only happens once. |
| Asked to run the pipeline live | Decline politely: "Random Forest training takes 30 minutes on this laptop — the trained models are saved and the dashboard is reading their real output." |
| Asked to show the code | `src/threshold_optimization.py` and `src/rule_baseline.py` are the two most impressive files. |

---

## 5. Viva questions and answers

### On method

**"Why only two models? Why not XGBoost or a neural network?"**
Scope was fixed at two for a two-member team, and the result justifies it: the
Random Forest already reaches the ceiling the data allows — a hand-written rule
gets precision 1.0000 on the same split. A third model could not beat that, so
the time went into threshold selection, ablation and verification instead,
which changed the outcome far more.

**"Why a chronological split instead of a random one?"**
A random split would put transactions from step 700 in training and step 100 in
test — training on the future. That is impossible in deployment and inflates
every score. The cost is that our test window has a higher fraud rate than
training, which we state explicitly rather than hide.

**"Why not SMOTE for the imbalance?"**
SMOTE would fabricate millions of synthetic financial records by interpolating
between real ones, which can produce impossible balance arithmetic — a
transaction whose amount does not match the balance change. Class weighting
achieves the same cost-sensitivity without inventing data, and Spark MLlib
supports it natively through `weightCol`.

**"Why is your threshold 0.67 and not 0.5?"**
0.5 assumes balanced classes and equal error costs. Neither holds. We measured
about 100 thresholds on the validation split and chose the one maximising F2,
which weights recall twice as heavily as precision — because a missed
fraudulent transfer is a direct financial loss while a false alarm costs an
analyst a few minutes. The test split played no part in that choice.

**"How do you know you haven't leaked the target?"**
Three ways. `isFlaggedFraud` was dropped in preprocessing — it is another
detector's output. `step` is used only to order the split, never as a feature.
And every remaining feature is computed from values recorded at transaction
time. The integration suite checks that the splits are strictly time-ordered.

### On results

**"99.95 % looks too good. Is it real?"**
It is real, and it is also not impressive — we tested that. A hand-written rule
with no machine learning at all (`TRANSFER or CASH_OUT` **and**
`amount == oldbalanceOrg` **and** `newbalanceOrig == 0`) reaches precision
1.0000 and recall 0.9730 on the same test data. PaySim's fraud agents drain
accounts exactly, so the pattern is nearly deterministic. The forest is
rediscovering it. These numbers describe the simulator, not real fraud.
*(Offer to run `python src/rule_baseline.py` if they want to see it.)*

**"Why is Logistic Regression so much worse?"**
Because the fraud signature is a *conjunction* of three conditions. A decision
tree represents that in three splits. Logistic regression adds up weighted
features and cannot express an AND at all — it is a structural limitation of
the model class, not a tuning failure. That is the clearest lesson in the
project.

**"Your test precision is higher than your validation precision. Isn't that backwards?"**
It is a prevalence effect. Fraud is 0.4361 % of the test window but 0.0575 % of
validation. Precision and PR-AUC both depend on how common the positive class
is; recall does not. Comparing recall across the splits is the fair comparison.

**"What is PR-AUC and why not ROC-AUC?"**
PR-AUC is the area under the precision-recall curve. ROC-AUC divides false
positives by the 6.35 million legitimate transactions, so it stays flattering
even for a poor model — our Logistic Regression scores 0.9939 ROC-AUC while
catching only 67 % of fraud. PR-AUC ignores true negatives and is the honest
summary for a rare positive class.

### On Big Data

**"Where is the 'Big Data' here? 557 MB is not big."**
Three things. The volume genuinely exceeds comfortable single-machine
processing in pandas, and the raw dataset is larger. More importantly the
*method* is distributed: Spark reads 16 part files in parallel, trains MLlib
models across cores on all 4.46 million training rows, and spills to disk
rather than failing. The same code runs unchanged on a cluster against a
dataset a hundred times larger — that is the point of the exercise.

**"Why Parquet instead of CSV?"**
CSV is text: every read re-parses 6.36 million lines and guesses types. Parquet
is columnar, typed and compressed, so repeated reads — training, tuning,
evaluation, dashboard aggregation — are several times faster and type-safe. The
conversion is verified: same row count, same fraud count, unique IDs.

**"How does the dashboard stay fast?"**
It never touches the full dataset. Streamlit re-runs its script on every click,
so Spark precomputes every analytic once into small aggregate files — 3.5 MB
total — and the dashboard reads only those, cached on file modification time.

### On engineering

**"What was the hardest part?"**
Honestly, the environment: this Mac had no Java at all, and Spark is a JVM
application. Homebrew no longer builds for Intel Macs, so we installed the
Temurin JDK from the official archive into the home directory and pinned Python
3.11, because PySpark 4.3.0 has no wheels for 3.14. That is all documented in
`docs/SETUP.md`.

**"Did your tests actually find anything?"**
Three real defects. Feature names were misaligned with the feature vector, so
coefficients would have been reported against the wrong features. The risk
bands collapsed so that every alert was "High Risk". And the Streamlit cache
key used an underscore-prefixed argument, which Streamlit ignores when
hashing — the dashboard would have shown stale numbers during this very demo.

**"What would you do differently?"**
Validate on a second, less deterministic dataset. PaySim's fraud is so
consistent that it cannot distinguish a good model from a mediocre one — the
ceiling is reached by a three-condition rule. A harder dataset would make the
model comparison meaningful.

---

## 6. Division of labour for the presentation

| Section | Presenter |
|---|---|
| Slides 1–6 (problem, dataset, architecture, Spark) | Member 1 |
| Slides 7–13 (features, split, models, results, honesty checks) | Member 2 |
| Slides 14–16 + live demo (alerts, dashboard, testing) | Member 2 |
| Slides 17–18 (conclusion, future work) | Member 1 |
| Questions | Whoever owns the area; hand over explicitly rather than guessing |

---

## 7. One-sentence summaries to have ready

- **The project:** "A Spark-based fraud detection system that processes 6.36
  million transactions, trains two MLlib models, and generates banded fraud
  alerts through an interactive dashboard."
- **The result:** "Random Forest catches 4,004 of 4,006 frauds with two false
  alarms — but we proved a hand-written rule does nearly as well, so that score
  describes the simulator, not real fraud."
- **The lesson:** "Choosing the decision threshold changed our results more
  than choosing the model."
