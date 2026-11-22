"""
FraudShield - Module 6: final model comparison on the TEST split.

Usage
-----
    python src/model_comparison.py

This is the ONLY script that touches the test data, and it runs after the
thresholds have already been fixed on the validation split by
src/threshold_optimization.py. That ordering is what keeps the test numbers
honest: nothing about the test split influenced the model or the cut-off.

Outputs
-------
    outputs/model_results/model_comparison.json   full detail
    outputs/model_results/model_comparison.csv    the comparison table
    outputs/model_results/confusion_matrices.json per-model confusion matrices

Every metric is computed for the FRAUD class as the positive class:

    precision = TP / (TP + FP)      how many of our alerts were real fraud
    recall    = TP / (TP + FN)      how much of the real fraud we caught
    F1        = harmonic mean of the two
    F2        = recall weighted twice as heavily as precision
    PR-AUC    threshold-free summary, the main metric for a rare class
    ROC-AUC   secondary; it looks generous on imbalanced data
    accuracy  reported only to show why it must not be used here
"""

from __future__ import annotations

import csv
import json
import sys
import time

from pyspark.ml import PipelineModel

import config
import model_evaluation as mev
from threshold_optimization import MODELS, fbeta


def load_policy(tag: str) -> dict | None:
    path = config.MODEL_RESULTS_DIR / f"{tag}_threshold_policy.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_training_summary(tag: str) -> dict | None:
    path = config.MODEL_RESULTS_DIR / f"{tag}_training.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    log = config.setup_logging()
    config.ensure_dirs()
    spark = config.get_spark("FraudShield_Model_Comparison")
    started = time.time()

    try:
        df = config.read_engineered(spark)
        train, val, test = config.split_chronologically(df)
        test = test.cache()
        n_test = test.count()
        n_test_fraud = test.filter(f"{config.TARGET_COL} = 1").count()
        log.info(
            "TEST split: rows=%s  fraud=%s (%.4f %%)",
            f"{n_test:,}", f"{n_test_fraud:,}", 100.0 * n_test_fraud / n_test,
        )
        log.info(
            "Note: the test window (later simulation steps) has a higher fraud "
            "rate than training, because PaySim generates fewer legitimate "
            "transactions late in the run. Precision and PR-AUC are therefore "
            "not directly comparable between validation and test."
        )

        rows = []
        details = {}
        confusions = {}

        for key, (name, model_dir, tag) in MODELS.items():
            policy = load_policy(tag)
            if policy is None:
                log.warning("No threshold policy for %s - run threshold_optimization.py first", name)
                continue
            if not (model_dir / "metadata").exists():
                log.warning("Model not found at %s - skipping", model_dir)
                continue

            log.info("=" * 72)
            log.info("Evaluating %s on the test split", name)
            model = PipelineModel.load(str(model_dir))
            threshold = policy["selected_threshold"]

            preds = model.transform(test)
            metrics = mev.evaluate(preds, threshold=threshold)
            metrics["f2"] = round(fbeta(metrics["precision"], metrics["recall"], 2.0), 6)

            mev.log_confusion_matrix(metrics, f"{name} - TEST @ threshold {threshold:.3f}")

            # Metrics at the naive 0.5 cut-off, to show what threshold tuning bought.
            metrics_half = mev.metrics_at_thresholds(
                mev.with_fraud_probability(preds), [0.5]
            )[0]
            log.info(
                "For reference, the same model at the default 0.5 threshold: "
                "precision=%.4f recall=%.4f f1=%.4f alerts=%s",
                metrics_half["precision"], metrics_half["recall"], metrics_half["f1"],
                f"{metrics_half['alerts_raised']:,}",
            )

            training = load_training_summary(tag) or {}
            details[key] = {
                "model": name,
                "model_path": str(model_dir),
                "threshold": threshold,
                "threshold_policy": policy.get("selection_policy"),
                "trained_on": training.get("trained_on"),
                "train_rows": training.get("train_rows"),
                "hyperparameters": {
                    k: training.get(k)
                    for k in ("best_reg_param", "max_iter", "best_num_trees",
                              "best_max_depth", "subsampling_rate")
                    if training.get(k) is not None
                },
                "validation": {
                    "pr_auc": policy["validation_pr_auc"],
                    "roc_auc": policy["validation_roc_auc"],
                    **{k: policy["selected_validation_metrics"][k]
                       for k in ("precision", "recall", "f1", "f2", "tp", "fp", "fn", "tn")},
                },
                "test": metrics,
                "test_at_0.5": metrics_half,
                "risk_bands": policy["risk_bands"],
            }
            confusions[key] = {
                "model": name,
                "threshold": threshold,
                "test": {k: metrics[k] for k in ("tp", "fp", "fn", "tn")},
                "validation": {
                    k: policy["selected_validation_metrics"][k]
                    for k in ("tp", "fp", "fn", "tn")
                },
            }
            rows.append(
                {
                    "model": name,
                    "threshold": threshold,
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                    "test_f1": metrics["f1"],
                    "test_f2": metrics["f2"],
                    "test_pr_auc": metrics["pr_auc"],
                    "test_roc_auc": metrics["roc_auc"],
                    "test_tp": metrics["tp"],
                    "test_fp": metrics["fp"],
                    "test_fn": metrics["fn"],
                    "test_tn": metrics["tn"],
                    "test_alerts": metrics["alerts_raised"],
                    "test_accuracy": metrics["accuracy"],
                    "validation_pr_auc": policy["validation_pr_auc"],
                    "validation_recall": policy["selected_validation_metrics"]["recall"],
                    "validation_precision": policy["selected_validation_metrics"]["precision"],
                }
            )

        if not rows:
            log.error("Nothing to compare. Train the models and tune thresholds first.")
            return 1

        # --- the final model was chosen on validation, before this script ran ---
        selection_path = config.MODEL_RESULTS_DIR / "final_model_selection.json"
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
        if selection is None and len(rows) == 1:
            selection = {
                "selected_model": rows[0]["model"],
                "selection_basis": "only one model available",
            }

        # --- comparison table ---
        log.info("=" * 72)
        log.info("MODEL COMPARISON (fraud = positive class, TEST split)")
        header = (
            f"{'model':<26} {'thr':>6} {'precision':>10} {'recall':>8} "
            f"{'F1':>7} {'F2':>7} {'PR-AUC':>8} {'ROC-AUC':>8} "
            f"{'TP':>6} {'FP':>8} {'FN':>6}"
        )
        log.info(header)
        log.info("-" * len(header))
        for r in rows:
            log.info(
                "%-26s %6.3f %10.4f %8.4f %7.4f %7.4f %8.4f %8.4f %6d %8d %6d",
                r["model"], r["threshold"], r["test_precision"], r["test_recall"],
                r["test_f1"], r["test_f2"], r["test_pr_auc"], r["test_roc_auc"],
                r["test_tp"], r["test_fp"], r["test_fn"],
            )
        if selection:
            log.info("Final model (selected on validation): %s", selection["selected_model"])

        # --- save ---
        csv_path = config.MODEL_RESULTS_DIR / "model_comparison.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info("Saved %s", csv_path)

        mev.save_json(
            {
                "test_rows": n_test,
                "test_fraud_rows": n_test_fraud,
                "test_fraud_rate_pct": round(100.0 * n_test_fraud / n_test, 6),
                "split_boundaries": {
                    "train_max_step": config.TRAIN_MAX_STEP,
                    "validation_max_step": config.VAL_MAX_STEP,
                },
                "final_model_selection": selection,
                "comparison_table": rows,
                "models": details,
                "elapsed_seconds": round(time.time() - started, 1),
            },
            config.MODEL_RESULTS_DIR / "model_comparison.json",
        )
        mev.save_json(confusions, config.MODEL_RESULTS_DIR / "confusion_matrices.json")

        log.info("Model comparison finished in %.1f s", time.time() - started)
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
