"""
FraudShield - Module 6: model evaluation helpers.

Imported by the training scripts, the threshold-tuning script and the model
comparison script. Running this file directly does nothing useful.

Why not accuracy?
-----------------
Only 0.1291 % of the transactions are fraud. A model that predicts
"legitimate" for every single transaction is 99.87 % accurate and catches zero
fraud. Accuracy is therefore meaningless here, and every metric below is
computed for the FRAUD class as the positive class.

Definitions used throughout the project
---------------------------------------
With fraud = positive:

  TP (true positive)   a fraudulent transaction the model flagged.      Caught.
  FN (false negative)  a fraudulent transaction the model missed.       Costly.
  FP (false positive)  a legitimate transaction the model flagged.      False alarm.
  TN (true negative)   a legitimate transaction correctly left alone.

  precision = TP / (TP + FP)
      Of the transactions we flagged, what share really was fraud.
      Low precision means investigators waste time on false alarms.

  recall    = TP / (TP + FN)
      Of all the fraud that happened, what share did we catch.
      Low recall means fraud slipped through, which is the expensive failure.

  F1        = 2 * precision * recall / (precision + recall)
      The harmonic mean. It is only high when BOTH are reasonably high.
      This is the fraud-class F1, not the weighted average over both classes.
      (A weighted F1 on this dataset would be ~0.999 purely because of the
      6.35 million legitimate rows, which tells us nothing.)

  PR-AUC    area under the precision-recall curve
      Summarises the precision/recall trade-off across all thresholds. It is
      the right summary metric for a rare positive class, because it ignores
      the huge number of true negatives. Note that PR-AUC depends on how
      common fraud is in the evaluated data, so PR-AUC values from splits with
      different fraud rates are not directly comparable.

  ROC-AUC   reported as a secondary metric only. It looks flattering on
      imbalanced data because the false-positive rate is divided by the very
      large number of legitimate transactions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config

LOG = logging.getLogger("fraudshield.evaluation")

FRAUD_PROB_COL = "fraud_probability"


def with_fraud_probability(
    predictions: DataFrame,
    probability_col: str = "probability",
    out_col: str = FRAUD_PROB_COL,
) -> DataFrame:
    """
    Pull the fraud probability out of MLlib's `probability` vector.

    MLlib returns a vector [P(legitimate), P(fraud)]; element 1 is the
    probability of the fraud class.
    """
    if out_col in predictions.columns:
        return predictions
    return predictions.withColumn(
        out_col, vector_to_array(F.col(probability_col))[1].cast("double")
    )


def metrics_at_thresholds(
    predictions: DataFrame,
    thresholds: list[float],
    prob_col: str = FRAUD_PROB_COL,
    label_col: str = config.TARGET_COL,
) -> list[dict]:
    """
    Confusion matrix and fraud-class metrics for many thresholds in ONE pass
    over the data.

    Scanning 6.36 million rows once per threshold would be wasteful, so all
    thresholds are evaluated as separate aggregate expressions in a single
    Spark job.
    """
    df = with_fraud_probability(predictions, out_col=prob_col)

    agg_exprs = []
    for i, t in enumerate(thresholds):
        flagged = F.col(prob_col) >= F.lit(t)
        is_fraud = F.col(label_col) == 1
        agg_exprs += [
            F.sum(F.when(flagged & is_fraud, 1).otherwise(0)).alias(f"tp_{i}"),
            F.sum(F.when(flagged & ~is_fraud, 1).otherwise(0)).alias(f"fp_{i}"),
            F.sum(F.when(~flagged & is_fraud, 1).otherwise(0)).alias(f"fn_{i}"),
            F.sum(F.when(~flagged & ~is_fraud, 1).otherwise(0)).alias(f"tn_{i}"),
        ]

    row = df.agg(*agg_exprs).collect()[0]

    results = []
    for i, t in enumerate(thresholds):
        tp, fp, fn, tn = (
            int(row[f"tp_{i}"]),
            int(row[f"fp_{i}"]),
            int(row[f"fn_{i}"]),
            int(row[f"tn_{i}"]),
        )
        results.append(_confusion_metrics(tp, fp, fn, tn, threshold=float(t)))
    return results


def _confusion_metrics(tp: int, fp: int, fn: int, tn: int, threshold: float) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    total = tp + fp + fn + tn
    return {
        "threshold": round(threshold, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        # Accuracy is reported ONLY to show how misleading it is here.
        "accuracy": round((tp + tn) / total, 8) if total else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 8) if (fp + tn) else 0.0,
        "alerts_raised": tp + fp,
        "alert_rate_pct": round(100.0 * (tp + fp) / total, 6) if total else 0.0,
    }


def area_metrics(
    predictions: DataFrame,
    probability_col: str = "probability",
    label_col: str = config.TARGET_COL,
) -> dict:
    """PR-AUC and ROC-AUC, which are threshold-independent."""
    pr = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderPR"
    ).evaluate(predictions)
    roc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderROC"
    ).evaluate(predictions)
    return {"pr_auc": round(float(pr), 6), "roc_auc": round(float(roc), 6)}


def evaluate(
    predictions: DataFrame,
    threshold: float,
    probability_col: str = "probability",
    label_col: str = config.TARGET_COL,
) -> dict:
    """Full evaluation of one model at one chosen threshold."""
    df = with_fraud_probability(predictions, probability_col=probability_col)
    df.cache()
    try:
        result = metrics_at_thresholds(df, [threshold], label_col=label_col)[0]
        result.update(area_metrics(df, probability_col=probability_col, label_col=label_col))
        return result
    finally:
        df.unpersist()


def default_threshold_grid() -> list[float]:
    """
    Thresholds to sweep during tuning.

    The grid is dense near 0 and near 1 because a class-weighted model pushes
    probabilities towards the middle, and because the interesting
    precision/recall trade-offs for rare events sit at the extremes.
    """
    coarse = [i / 100 for i in range(1, 100)]
    fine_low = [0.001, 0.002, 0.005]
    fine_high = [0.995, 0.998, 0.999]
    return sorted(set(fine_low + coarse + fine_high))


def log_confusion_matrix(metrics: dict, title: str = "") -> None:
    """Print a readable 2x2 confusion matrix with the fraud-class metrics."""
    if title:
        LOG.info("--- %s ---", title)
    LOG.info("threshold = %.4f", metrics["threshold"])
    LOG.info("                     predicted")
    LOG.info("                 legit        fraud")
    LOG.info("actual legit  %10s   %10s   <- FP are false alarms",
             f"{metrics['tn']:,}", f"{metrics['fp']:,}")
    LOG.info("actual fraud  %10s   %10s   <- FN are missed fraud",
             f"{metrics['fn']:,}", f"{metrics['tp']:,}")
    LOG.info(
        "precision=%.4f  recall=%.4f  f1=%.4f  (fraud class)",
        metrics["precision"], metrics["recall"], metrics["f1"],
    )
    if "pr_auc" in metrics:
        LOG.info("PR-AUC=%.4f  ROC-AUC=%.4f", metrics["pr_auc"], metrics["roc_auc"])
    LOG.info(
        "accuracy=%.6f (high only because %s of %s rows are legitimate)",
        metrics["accuracy"], f"{metrics['tn'] + metrics['fp']:,}", f"{metrics['total']:,}",
    )


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    LOG.info("Saved %s", path)
