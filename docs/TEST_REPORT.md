# FraudShield — Test Report

**Scope:** end-to-end verification of the integrated system (Member 1's data
engineering output + Member 2's ML, alerting and dashboard layers).

**How to reproduce:**

```bash
source env.sh && python tests/test_integration.py
```

The suite exits non-zero if any check fails, so it works as a single
"is the project healthy?" command before a demonstration.

---

## 1. Testing strategy

The project is a data pipeline, so unit-testing individual functions against
mock inputs would prove very little: the realistic failure modes are a schema
that drifts, a split that leaks, a model that cannot be reloaded, a metric
computed on the wrong class, or a dashboard reading a stale file. Every check
below therefore runs against the **real 6.36-million-row dataset and the real
saved artefacts**, not fixtures.

Three categories:

| Category | Checks | What it protects against |
|---|---|---|
| Data integrity | 1–5 | Wrong dataset, silent schema change, leaking or empty splits |
| Model correctness | 6–12 | Broken pipeline, unusable saved model, wrong metric arithmetic |
| Application readiness | 13–16 | Missing policy, malformed alerts, dashboard that cannot start |

Two checks deserve particular mention:

- **Check 11 (save/reload fidelity)** saves the fitted pipeline to a temporary
  directory, reloads it, and asserts that predictions on 2,000 rows are
  *identical*. This is what proves the saved model is genuinely deployable
  rather than merely present on disk.
- **Check 12 (metric arithmetic)** runs the evaluation code over a ten-row
  hand-worked example with a known answer (TP=3, FP=2, FN=1, TN=4 →
  precision 0.6, recall 0.75, F1 0.6667). If the fraud class were ever swapped
  with the legitimate class, or a weighted F1 reported in place of the
  fraud-class F1, this check fails immediately.

---

## 2. Results

**16 passed, 0 failed.** Run on the full dataset; console output reproduced
verbatim below.

```
FraudShield integration tests
========================================================================
  PASS  1  engineered dataset loads in Spark  [source=Parquet]
  PASS  2  schema has the expected engineered columns  [13 expected columns present, 14 total]
  PASS  3  row count is 6,362,620  [6,362,620 rows]
  PASS  4  fraud distribution is 8,213 fraud / 6,354,407 legitimate  [8,213 fraud (0.1291%)]
  PASS  5  chronological split is time-ordered and non-empty  [train 4,463,587 (<= step 323), val 980,416 (564 fraud), test 918,617 (4006 fraud)]
  PASS  6  feature pipeline executes and produces a feature vector  [vector size 17, names aligned]
  PASS  7  class weights are computed from the training split  [legit=0.5004, fraud=612.6]
  PASS  8  saved Logistic Regression model loads  [LogisticRegression (threshold=0.980, High>=0.999, Medium>=0.990)]
  PASS  9  saved Random Forest model loads  [RandomForestClassifier (threshold=0.670, High>=1.000, Medium>=1.000)]
  PASS  10 reloaded models produce valid probabilities  [lr: 0.000-1.000; rf: 0.000-1.000]
  PASS  11 a model can be saved, reloaded and give identical predictions  [2000 rows identical (max diff 0.00e+00)]
  PASS  12 evaluation metrics are mathematically correct  [TP=3 FP=2 FN=1 TN=4, precision=0.6, recall=0.75, F1=0.6667]
  PASS  13 threshold policy files are present and consistent  [logistic_regression@0.980, random_forest@0.670]
  PASS  14 fraud alerts exist and respect the operating threshold  [4,006 exported, 4,006 total, threshold 0.670]
  PASS  15 dashboard data files are present and readable  [9 files, 3580 KB total]
  PASS  16 the Streamlit dashboard imports and compiles  [streamlit 1.42.0, app.py compiles]
========================================================================
16 passed, 0 failed, 16 checks total
```

---

## 3. Check-by-check rationale

| # | Check | Why it exists |
|---|---|---|
| 1 | Dataset loads in Spark | Catches a broken path, a missing symlink or a corrupt Parquet write |
| 2 | Expected schema present | Catches a renamed or dropped engineered column |
| 3 | Row count = 6,362,620 | Catches a partial read of the 16 part files |
| 4 | Fraud distribution exact | Catches the wrong dataset version being used |
| 5 | Splits time-ordered, non-empty, enough fraud | **Catches data leakage** — train must end before validation begins |
| 6 | Feature pipeline produces an aligned vector | Catches the name/vector misalignment that would mislabel coefficients |
| 7 | Class weights from the training split only | Catches weights computed over the whole dataset (leakage) |
| 8 | Logistic Regression loads | Catches an incomplete model write |
| 9 | Random Forest loads | As above |
| 10 | Probabilities valid and risk categories legal | Catches NaNs, out-of-range scores, unexpected category strings |
| 11 | Save → reload → identical predictions | Proves deployability |
| 12 | Metric arithmetic against hand-worked values | Catches a positive-class or averaging mistake |
| 13 | Threshold policies consistent and ordered | Catches `High < Medium < operating` band inversion |
| 14 | Alerts respect the threshold; Parquet complete | Catches alerts exported below the operating threshold |
| 15 | Dashboard aggregates present and agreeing with the dataset | Catches stale or partial aggregate files |
| 16 | Dashboard imports and compiles | Catches a syntax or import error before the demo |

---

## 4. Defects found and fixed during testing

Testing was not a formality — it surfaced three real defects, all fixed:

| Defect | Symptom | Impact if shipped | Fix |
|---|---|---|---|
| Feature-name/vector misalignment | 16 names generated for a 17-wide vector (`OneHotEncoder` with `handleInvalid="keep"` reserves an extra slot) | Coefficients and feature importances reported against the **wrong feature names** | Names derived from the true vector width and padded explicitly |
| Degenerate risk bands | Random Forest exceeds 90 % precision at its operating threshold, so all three precision-based bands collapsed to one value | **Every alert labelled "High Risk"** — the risk categorisation would have been meaningless | Degeneracy detected; bands fall back to percentiles of the flagged score distribution, with the fallback recorded in the policy file |
| Streamlit cache never invalidated | Cache-busting argument was named `_stamp`; Streamlit ignores underscore-prefixed arguments when building cache keys | Dashboard would show **stale numbers indefinitely** after a pipeline re-run — including during the live demonstration | Parameter renamed to `stamp`; a comment in the source records why |

Two further issues were caught by the scripts' own validation: a
`Decimal`-not-JSON-serialisable error from Spark SQL's `ROUND`, and a missing
`pyspark.sql.functions` import in the threshold module.

---

## 5. Manual verification

| Item | Method | Result |
|---|---|---|
| Dashboard renders real data | Opened all five sections in a browser | Pass — values match `outputs/` files |
| Band fix visible end-to-end | Re-ran alerts, confirmed 489 / 1,623 / 1,894 split in the UI | Pass |
| Download buttons | Clicked through the Downloads section | Pass — all present files offered with sizes |
| Member 1's scripts untouched | `git diff` against commit 7ad74e5 for `src/` | Pass — no modifications |
| `test_spark.py` (Member 1's) still runs | Executed unmodified | Pass — Spark 4.3.0 session created |

---

## 6. Known gaps in test coverage

Stated rather than hidden:

- **No unit tests for individual Spark transformations.** The integration suite
  tests behaviour end-to-end; a transformation could be wrong in a way the
  aggregate checks do not reveal.
- **No performance or load testing.** Runtimes are recorded (Random Forest:
  1,785.8 s) but there is no benchmark suite or scaling study.
- **No automated UI testing.** Dashboard verification is manual; check 16 only
  confirms the app imports and compiles.
- **Single-environment testing.** All checks ran on macOS with Spark in local
  mode. The code is not verified on a real cluster or on Windows.
