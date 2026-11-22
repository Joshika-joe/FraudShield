"""
FraudShield - Module 7: fraud threshold optimisation (validation split only).

Usage
-----
    python src/threshold_optimization.py              # both saved models
    python src/threshold_optimization.py --model lr   # one model only

Why 0.5 is the wrong threshold here
-----------------------------------
A classifier outputs a probability. Turning that probability into a decision
needs a cut-off. The usual default, 0.5, assumes the two classes are equally
common and equally costly. Here neither is true: fraud is 0.1291 % of the
data, and the models are trained with class weights that deliberately push
fraud probabilities upwards. At 0.5 a weighted model flags far too much.

So the cut-off is chosen by measuring precision and recall at ~100 candidate
thresholds on the VALIDATION split, and picking the one that best matches the
project's objective. The test split is never used here.

Choosing the operating point
----------------------------
Four candidate policies are measured and recorded:

  best_f1   threshold with the highest fraud-class F1
            (treats a missed fraud and a false alarm as equally bad)

  best_f2   threshold with the highest F-beta score with beta = 2
            F2 = 5*P*R / (4*P + R), which weights recall twice as heavily
            as precision

  recall_at_least_0.80   of all thresholds reaching 80 % recall, the one with
                         the highest precision

  precision_at_least_0.50  of all thresholds reaching 50 % precision, the one
                           with the highest recall

FraudShield uses **best_f2** as its operating threshold. Reasoning: an
undetected fraudulent transfer is a direct financial loss, while a false alarm
costs an analyst a few minutes of review. The two errors are not equally
expensive, so a metric that favours recall is the honest choice - but F2 still
penalises a threshold that floods the team with alerts, which is why it is
preferred over simply maximising recall (recall is trivially 1.0 at a
threshold of 0, where every transaction is an "alert" and nothing is learned).

Risk-category policy (used by Module 8)
---------------------------------------
Alerts are banded by the precision actually measured on the validation split,
not by invented numbers:

  High Risk    probability >= the lowest threshold reaching 90 % precision
  Medium Risk  probability >= the lowest threshold reaching 50 % precision
  Low Risk     probability >= the operating threshold (but below Medium)

If a precision level is unreachable on validation, the band falls back to a
quantile of the flagged probabilities and the fallback is recorded in the
saved policy file so it is visible in the report rather than hidden.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

import config
import model_evaluation as mev

MODELS = {
    "lr": ("LogisticRegression", config.LR_MODEL_DIR, "logistic_regression"),
    "rf": ("RandomForestClassifier", config.RF_MODEL_DIR, "random_forest"),
}


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def pick_policies(sweep: list[dict]) -> dict:
    """Choose candidate operating thresholds from the measured sweep."""
    usable = [m for m in sweep if m["tp"] > 0]
    if not usable:
        raise RuntimeError("No threshold produced a single true positive")

    best_f1 = max(usable, key=lambda m: m["f1"])
    best_f2 = max(usable, key=lambda m: fbeta(m["precision"], m["recall"], 2.0))

    recall_80 = [m for m in usable if m["recall"] >= 0.80]
    precision_50 = [m for m in usable if m["precision"] >= 0.50]

    policies = {
        "best_f1": best_f1,
        "best_f2": best_f2,
        "recall_at_least_0.80": (
            max(recall_80, key=lambda m: m["precision"]) if recall_80 else None
        ),
        "precision_at_least_0.50": (
            max(precision_50, key=lambda m: m["recall"]) if precision_50 else None
        ),
    }
    for name, m in policies.items():
        if m is not None:
            m = dict(m)
            m["f2"] = round(fbeta(m["precision"], m["recall"], 2.0), 6)
            policies[name] = m
    return policies


def pick_risk_bands(
    sweep: list[dict],
    operating_threshold: float,
    val_preds=None,
) -> dict:
    """
    Derive the High / Medium / Low probability bands from validation precision.

    `val_preds` (validation rows with a fraud_probability column) is used only
    for the degenerate case described at the bottom of this function.
    """
    bands: dict = {"operating_threshold": operating_threshold, "fallbacks": []}

    def lowest_threshold_with_precision(target: float):
        candidates = [
            m for m in sweep
            if m["precision"] >= target and m["tp"] > 0 and m["threshold"] >= operating_threshold
        ]
        return min(candidates, key=lambda m: m["threshold"]) if candidates else None

    high = lowest_threshold_with_precision(0.90)
    if high is None:
        # Cannot reach 90 % precision: fall back to the top of the sweep.
        high = max(sweep, key=lambda m: (m["precision"], m["threshold"]))
        bands["fallbacks"].append(
            "90% precision unreachable on validation; High Risk band uses the "
            "highest-precision threshold measured instead"
        )
    medium = lowest_threshold_with_precision(0.50)
    if medium is None:
        medium = high
        bands["fallbacks"].append(
            "50% precision unreachable on validation; Medium Risk band set equal "
            "to the High Risk band"
        )

    bands["high_risk_threshold"] = high["threshold"]
    bands["high_risk_validation"] = {
        "precision": high["precision"], "recall": high["recall"], "tp": high["tp"], "fp": high["fp"],
    }
    bands["medium_risk_threshold"] = medium["threshold"]
    bands["medium_risk_validation"] = {
        "precision": medium["precision"], "recall": medium["recall"], "tp": medium["tp"], "fp": medium["fp"],
    }
    bands["low_risk_threshold"] = operating_threshold
    bands["basis"] = "validation precision levels (90% / 50%)"

    # --- degenerate case -------------------------------------------------
    # A very accurate model is already above 90 % precision at its operating
    # threshold, so "lowest threshold reaching 90 % precision" and "lowest
    # reaching 50 %" both collapse onto the operating threshold, and every
    # alert would be labelled High Risk - which tells an analyst nothing.
    #
    # When that happens the bands fall back to the shape of the score
    # distribution itself: the most confident alerts are High, the middle band
    # Medium, the rest Low. This is still derived from measured validation
    # data, and the fallback is recorded so it is visible in the report rather
    # than hidden.
    degenerate = (
        bands["high_risk_threshold"] <= operating_threshold + 1e-9
        and bands["medium_risk_threshold"] <= operating_threshold + 1e-9
    )
    if degenerate and val_preds is not None:
        flagged = val_preds.filter(F.col(mev.FRAUD_PROB_COL) >= operating_threshold)
        n_flagged = flagged.count()
        if n_flagged >= 20:
            q_medium, q_high = flagged.approxQuantile(
                mev.FRAUD_PROB_COL, [0.50, 0.85], 0.001
            )
            # Keep the bands strictly ordered and above the operating point.
            q_medium = max(q_medium, operating_threshold)
            q_high = max(q_high, q_medium)
            if q_high > q_medium > operating_threshold or q_high > operating_threshold:
                bands["high_risk_threshold"] = round(float(q_high), 6)
                bands["medium_risk_threshold"] = round(float(q_medium), 6)
                bands["basis"] = (
                    "quantiles of the predicted fraud probability among flagged "
                    "validation transactions (85th / 50th percentile)"
                )
                bands["fallbacks"].append(
                    "The model already exceeds 90 % precision at its operating "
                    "threshold, so precision-based bands all collapsed onto that "
                    "one value and every alert would have been High Risk. The "
                    "bands are therefore taken from the 85th and 50th percentiles "
                    f"of the {n_flagged:,} flagged validation scores instead."
                )
    return bands


def run_for_model(spark, key: str, log) -> dict | None:
    name, model_dir, tag = MODELS[key]
    if not (model_dir / "metadata").exists():
        log.warning("%s not found at %s - train it first, skipping", name, model_dir)
        return None

    log.info("=" * 72)
    log.info("Threshold sweep for %s", name)
    log.info("Loading saved pipeline from %s", model_dir)
    model = PipelineModel.load(str(model_dir))

    df = config.read_engineered(spark)
    _, val, _ = config.split_chronologically(df)
    val_preds = mev.with_fraud_probability(model.transform(val))
    val_preds = val_preds.select(mev.FRAUD_PROB_COL, config.TARGET_COL).cache()
    n_val = val_preds.count()
    n_fraud = val_preds.filter(f"{config.TARGET_COL} = 1").count()
    log.info("Validation rows=%s  fraud=%s", f"{n_val:,}", f"{n_fraud:,}")

    thresholds = mev.default_threshold_grid()
    t0 = time.time()
    sweep = mev.metrics_at_thresholds(val_preds, thresholds)
    for m in sweep:
        m["f2"] = round(fbeta(m["precision"], m["recall"], 2.0), 6)
    log.info("Swept %d thresholds in %.1f s (single pass over the data)",
             len(thresholds), time.time() - t0)

    areas = mev.area_metrics(model.transform(val), probability_col="probability")
    policies = pick_policies(sweep)

    log.info("Candidate operating points on validation:")
    for pname, m in policies.items():
        if m is None:
            log.info("   %-26s not achievable on this validation split", pname)
            continue
        log.info(
            "   %-26s threshold=%.3f  precision=%.4f  recall=%.4f  f1=%.4f  f2=%.4f  alerts=%s",
            pname, m["threshold"], m["precision"], m["recall"], m["f1"], m["f2"],
            f"{m['alerts_raised']:,}",
        )

    selected = policies["best_f2"]
    bands = pick_risk_bands(sweep, selected["threshold"], val_preds=val_preds)

    log.info("SELECTED operating threshold = %.3f (policy: best_f2)", selected["threshold"])
    mev.log_confusion_matrix(selected, f"{name} - validation @ selected threshold")
    log.info("Risk bands: High >= %.6f, Medium >= %.6f, Low >= %.6f",
             bands["high_risk_threshold"], bands["medium_risk_threshold"],
             bands["low_risk_threshold"])
    for note in bands["fallbacks"]:
        log.warning("Risk band fallback: %s", note)

    # --- save the sweep as CSV (used by the dashboard's threshold chart) ---
    sweep_path = config.MODEL_RESULTS_DIR / f"{tag}_threshold_sweep.csv"
    with sweep_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    log.info("Saved %s", sweep_path)

    result = {
        "model": name,
        "model_key": key,
        "model_path": str(model_dir),
        "validation_rows": n_val,
        "validation_fraud_rows": n_fraud,
        "validation_pr_auc": areas["pr_auc"],
        "validation_roc_auc": areas["roc_auc"],
        "selection_policy": "best_f2",
        "selected_threshold": selected["threshold"],
        "selected_validation_metrics": selected,
        "candidate_policies": policies,
        "risk_bands": bands,
        "threshold_grid_size": len(thresholds),
    }
    mev.save_json(result, config.MODEL_RESULTS_DIR / f"{tag}_threshold_policy.json")
    val_preds.unpersist()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune the fraud threshold on validation data")
    parser.add_argument("--model", choices=["lr", "rf", "both"], default="both")
    args = parser.parse_args()

    log = config.setup_logging()
    config.ensure_dirs()
    spark = config.get_spark("FraudShield_Threshold_Optimization")

    try:
        keys = ["lr", "rf"] if args.model == "both" else [args.model]
        results = {}
        for key in keys:
            r = run_for_model(spark, key, log)
            if r:
                results[key] = r

        if not results:
            log.error("No trained models were found. Run the training scripts first.")
            return 1

        # --- model selection, on VALIDATION only ---
        if len(results) > 1:
            best_key = max(results, key=lambda k: results[k]["validation_pr_auc"])
            log.info("=" * 72)
            log.info("Model selection (validation PR-AUC, test split untouched):")
            for k, r in results.items():
                log.info("   %-26s PR-AUC=%.6f  F2=%.4f  threshold=%.3f",
                         r["model"], r["validation_pr_auc"],
                         r["selected_validation_metrics"]["f2"], r["selected_threshold"])
            log.info("Selected final model: %s", results[best_key]["model"])
            mev.save_json(
                {
                    "selected_model_key": best_key,
                    "selected_model": results[best_key]["model"],
                    "selection_basis": "highest validation PR-AUC (test split not used)",
                    "selected_threshold": results[best_key]["selected_threshold"],
                    "risk_bands": results[best_key]["risk_bands"],
                    "validation_comparison": {
                        k: {
                            "pr_auc": r["validation_pr_auc"],
                            "roc_auc": r["validation_roc_auc"],
                            "threshold": r["selected_threshold"],
                            "precision": r["selected_validation_metrics"]["precision"],
                            "recall": r["selected_validation_metrics"]["recall"],
                            "f1": r["selected_validation_metrics"]["f1"],
                            "f2": r["selected_validation_metrics"]["f2"],
                        }
                        for k, r in results.items()
                    },
                },
                config.MODEL_RESULTS_DIR / "final_model_selection.json",
            )
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
