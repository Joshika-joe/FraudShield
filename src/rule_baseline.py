"""
FraudShield - deterministic rule baseline (a sanity check on the ML results).

Usage
-----
    python src/rule_baseline.py

Why this script exists
----------------------
The Random Forest scored almost perfectly on the test split (F1 ~ 0.9995). A
result that good is usually a warning sign, not a success, so before reporting
it we checked whether the dataset is simply near-deterministic.

This script evaluates a hand-written rule that uses **no machine learning at
all**:

    type is TRANSFER or CASH_OUT
    AND amount == oldbalanceOrg      (the transfer takes the whole balance)
    AND newbalanceOrig == 0          (the sender's account is left empty)

In other words: "someone emptied this account in one move." PaySim's fraud
agents do exactly that, which leaves an almost perfect signature in the balance
columns.

Interpreting the output
-----------------------
If the rule alone reaches very high precision and recall, then the models are
not performing a subtle feat - they are rediscovering a deterministic pattern
built into the simulator. That is important to state honestly in the report,
because it means the near-perfect scores describe PaySim, not real-world fraud
detection difficulty.

It also explains the gap between the two models: the rule is a CONJUNCTION of
three conditions. A decision tree expresses it in three splits. Logistic
Regression adds up weighted features and cannot represent an AND of conditions
at all, which is why it plateaus far below the forest.

Note this is NOT target leakage: the rule uses only transaction fields recorded
at the time of the transaction. Nothing derived from `isFraud` or
`isFlaggedFraud` is involved.

Output: outputs/model_results/rule_baseline.json
"""

from __future__ import annotations

import sys

from pyspark.sql import functions as F

import config
import model_evaluation as mev


def rule_column():
    """The hand-written 'account drained in one move' rule."""
    return (
        F.col("type").isin("TRANSFER", "CASH_OUT")
        & (F.abs(F.col("amount") - F.col("oldbalanceOrg")) < 0.01)
        & (F.col("newbalanceOrig") == 0)
    )


def evaluate_rule(df, name: str, log) -> dict:
    counts = df.select(
        rule_column().alias("flagged"), F.col(config.TARGET_COL).alias("label")
    ).agg(
        F.sum(F.when(F.col("flagged") & (F.col("label") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when(F.col("flagged") & (F.col("label") == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when(~F.col("flagged") & (F.col("label") == 1), 1).otherwise(0)).alias("fn"),
        F.sum(F.when(~F.col("flagged") & (F.col("label") == 0), 1).otherwise(0)).alias("tn"),
    ).collect()[0]

    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result = {
        "split": name,
        "rows": tp + fp + fn + tn,
        "fraud_rows": tp + fn,
        "rule_matches": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }
    log.info(
        "%-11s rows=%-10s fraud=%-6s rule matched=%-6s  precision=%.4f  recall=%.4f  F1=%.4f",
        name, f"{result['rows']:,}", f"{result['fraud_rows']:,}",
        f"{result['rule_matches']:,}", precision, recall, f1,
    )
    return result


def main() -> int:
    log = config.setup_logging()
    config.ensure_dirs()
    spark = config.get_spark("FraudShield_Rule_Baseline")

    try:
        df = config.read_engineered(spark).cache()
        train, val, test = config.split_chronologically(df)

        log.info("Rule: type in (TRANSFER, CASH_OUT) AND amount == oldbalanceOrg "
                 "AND newbalanceOrig == 0")
        log.info("No machine learning is involved in these numbers.")

        results = {
            "whole_dataset": evaluate_rule(df, "all", log),
            "train": evaluate_rule(train, "train", log),
            "validation": evaluate_rule(val, "validation", log),
            "test": evaluate_rule(test, "test", log),
        }

        test_r = results["test"]
        log.info("")
        log.info("INTERPRETATION")
        log.info(
            "On the test split this rule alone reaches precision %.4f and recall %.4f.",
            test_r["precision"], test_r["recall"],
        )
        if test_r["precision"] > 0.95 and test_r["recall"] > 0.9:
            log.info(
                "That is near-deterministic: PaySim's fraud agents drain the sender's "
                "account exactly, so the near-perfect model scores reflect the "
                "simulator's generation process, not the difficulty of real fraud "
                "detection. The trained models are rediscovering this conjunction."
            )
        log.info(
            "Logistic Regression cannot represent an AND of three conditions, which "
            "is why it scores far below the Random Forest on the same features."
        )

        mev.save_json(
            {
                "rule": ("type IN (TRANSFER, CASH_OUT) AND "
                         "abs(amount - oldbalanceOrg) < 0.01 AND newbalanceOrig = 0"),
                "uses_machine_learning": False,
                "uses_label_derived_features": False,
                "results": results,
                "interpretation": (
                    "The rule alone separates PaySim fraud almost perfectly, so the "
                    "models' near-perfect scores describe the simulator's "
                    "deterministic fraud pattern rather than real-world performance."
                ),
            },
            config.MODEL_RESULTS_DIR / "rule_baseline.json",
        )
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
