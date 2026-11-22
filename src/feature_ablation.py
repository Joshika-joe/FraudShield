"""
FraudShield - feature ablation: are the balance-mismatch features worth keeping?

Usage
-----
    python src/feature_ablation.py                 # Logistic Regression, full data
    python src/feature_ablation.py --models lr rf  # both models (RF is slow)
    python src/feature_ablation.py --sample 0.05   # quick version

Why this script exists
----------------------
Member 1's handover notes say: "Compare performance with and without the
balance mismatch features. Their usefulness is not guaranteed."

That is a fair warning. `sender_balance_mismatch` is raised on 79.8 % of all
transactions and `destination_balance_mismatch` on 65.8 %, partly because the
formula behind them (sender balance should fall by the amount, destination
balance should rise by it) does not apply to every transaction type - for
CASH_IN the sender's balance rises, so the sender flag is true by construction.
A feature that fires on four rows out of five may be adding noise, or may be
re-encoding information the `type` column already carries.

So instead of assuming, this script trains the same model twice on the same
chronological split - once with the two flags and once without - and compares
validation PR-AUC. Nothing else changes between the two runs, and the test
split is not touched.

Output: outputs/model_results/feature_ablation.json
"""

from __future__ import annotations

import argparse
import sys
import time

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.storagelevel import StorageLevel

import config
import ml_preprocessing as mlp
import model_evaluation as mev
from threshold_optimization import fbeta


def fit_and_score(spark, model_name, include_mismatch, train, val, log, args):
    """Train one configuration and return its validation metrics."""
    w_legit, w_fraud = mlp.compute_class_weights(train)
    train_w = mlp.add_class_weight_column(train, w_legit, w_fraud)

    scale = model_name == "lr"
    pipeline, features_col, numeric_cols = mlp.build_feature_pipeline(
        include_mismatch=include_mismatch, scale=scale
    )
    feature_model = pipeline.fit(train_w)
    train_feat = feature_model.transform(train_w).select(
        features_col, config.TARGET_COL, mlp.WEIGHT_COL
    ).persist(StorageLevel.MEMORY_AND_DISK)
    val_feat = feature_model.transform(val).select(
        features_col, config.TARGET_COL
    ).persist(StorageLevel.MEMORY_AND_DISK)
    train_feat.count()

    if model_name == "lr":
        estimator = LogisticRegression(
            featuresCol=features_col, labelCol=config.TARGET_COL,
            weightCol=mlp.WEIGHT_COL, maxIter=args.max_iter,
            regParam=args.reg_param, elasticNetParam=0.0, standardization=False,
        )
        params = {"reg_param": args.reg_param, "max_iter": args.max_iter}
    else:
        estimator = RandomForestClassifier(
            featuresCol=features_col, labelCol=config.TARGET_COL,
            weightCol=mlp.WEIGHT_COL, numTrees=args.num_trees,
            maxDepth=args.max_depth, maxBins=32, subsamplingRate=0.6,
            featureSubsetStrategy="sqrt", minInstancesPerNode=20,
            seed=config.RANDOM_SEED, cacheNodeIds=True, maxMemoryInMB=512,
        )
        params = {"num_trees": args.num_trees, "max_depth": args.max_depth}

    t0 = time.time()
    label = "with mismatch flags" if include_mismatch else "without mismatch flags"
    log.info("Fitting %s (%s), %d numeric features ...",
             model_name.upper(), label, len(numeric_cols))
    model = estimator.fit(train_feat)
    fit_seconds = time.time() - t0

    preds = model.transform(val_feat)
    areas = mev.area_metrics(preds)

    # Best achievable F2 on validation, so the comparison is not tied to one cut-off.
    sweep = mev.metrics_at_thresholds(
        mev.with_fraud_probability(preds), mev.default_threshold_grid()
    )
    usable = [m for m in sweep if m["tp"] > 0]
    best_f2 = max(usable, key=lambda m: fbeta(m["precision"], m["recall"], 2.0))
    best_f1 = max(usable, key=lambda m: m["f1"])

    train_feat.unpersist()
    val_feat.unpersist()

    result = {
        "model": model_name,
        "include_mismatch_features": include_mismatch,
        "n_numeric_features": len(numeric_cols),
        "hyperparameters": params,
        "fit_seconds": round(fit_seconds, 1),
        "validation_pr_auc": areas["pr_auc"],
        "validation_roc_auc": areas["roc_auc"],
        "best_f1_threshold": best_f1["threshold"],
        "best_f1": best_f1["f1"],
        "best_f2_threshold": best_f2["threshold"],
        "best_f2": round(fbeta(best_f2["precision"], best_f2["recall"], 2.0), 6),
        "best_f2_precision": best_f2["precision"],
        "best_f2_recall": best_f2["recall"],
    }
    log.info(
        "   PR-AUC=%.6f  best F1=%.4f  best F2=%.4f (precision=%.4f recall=%.4f)  [%.1f s]",
        result["validation_pr_auc"], result["best_f1"], result["best_f2"],
        result["best_f2_precision"], result["best_f2_recall"], fit_seconds,
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Balance-mismatch feature ablation")
    p.add_argument("--models", nargs="+", choices=["lr", "rf"], default=["lr"])
    p.add_argument("--sample", type=float, default=None)
    p.add_argument("--reg-param", type=float, default=0.0)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--num-trees", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=8)
    args = p.parse_args()

    log = config.setup_logging()
    config.ensure_dirs()
    spark = config.get_spark("FraudShield_Feature_Ablation")
    started = time.time()

    try:
        train, val, _ = mlp.load_splits(spark, sample_fraction=args.sample)
        train = train.cache()
        val = val.cache()
        log.info("Train rows=%s  Validation rows=%s", f"{train.count():,}", f"{val.count():,}")

        runs = []
        for model_name in args.models:
            for include in (True, False):
                runs.append(fit_and_score(spark, model_name, include, train, val, log, args))

        # --- verdict ---
        log.info("=" * 72)
        log.info("ABLATION RESULT (validation split, PR-AUC is the headline metric)")
        verdicts = {}
        for model_name in args.models:
            with_m = next(r for r in runs if r["model"] == model_name and r["include_mismatch_features"])
            without_m = next(r for r in runs if r["model"] == model_name and not r["include_mismatch_features"])
            delta = with_m["validation_pr_auc"] - without_m["validation_pr_auc"]
            rel = (100.0 * delta / without_m["validation_pr_auc"]
                   if without_m["validation_pr_auc"] else 0.0)
            keep = delta > 0
            verdicts[model_name] = {
                "pr_auc_with": with_m["validation_pr_auc"],
                "pr_auc_without": without_m["validation_pr_auc"],
                "absolute_change": round(delta, 6),
                "relative_change_pct": round(rel, 3),
                "f2_with": with_m["best_f2"],
                "f2_without": without_m["best_f2"],
                "verdict": ("keep the mismatch features" if keep
                            else "the mismatch features do not help"),
            }
            log.info(
                "%-4s  PR-AUC with=%.6f  without=%.6f  change=%+.6f (%+.2f %%)  ->  %s",
                model_name.upper(), with_m["validation_pr_auc"],
                without_m["validation_pr_auc"], delta, rel, verdicts[model_name]["verdict"],
            )

        mev.save_json(
            {
                "trained_on": "sample" if args.sample else "full_dataset",
                "sample_fraction": args.sample,
                "mismatch_features": config.MISMATCH_COLS,
                "runs": runs,
                "verdicts": verdicts,
                "note": (
                    "Both runs use the same chronological split, the same class "
                    "weights and the same hyperparameters; only the two mismatch "
                    "columns differ. The test split was not used."
                ),
                "elapsed_seconds": round(time.time() - started, 1),
            },
            config.MODEL_RESULTS_DIR / "feature_ablation.json",
        )
        log.info("Ablation finished in %.1f s", time.time() - started)
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
