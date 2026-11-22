"""
FraudShield - Model 1: Logistic Regression (Spark MLlib).

Usage
-----
    python src/train_logistic_regression.py                  # full dataset, tuned
    python src/train_logistic_regression.py --sample 0.02    # 2 % sample, fast debug
    python src/train_logistic_regression.py --no-tune        # single default fit
    python src/train_logistic_regression.py --no-mismatch    # ablation: drop the
                                                             # two mismatch flags

What this script does
---------------------
1. Reads the engineered dataset and splits it chronologically by `step`
   (train = earliest ~70 %, validation = next ~15 %, test = last ~15 %).
2. Computes 'balanced' class weights from the TRAINING split only.
3. Builds the MLlib feature pipeline WITH scaling (a linear model needs it).
4. Fits Logistic Regression, optionally trying a few regularisation values and
   keeping the one with the best validation PR-AUC.
5. Saves the fitted pipeline + model to models/logistic_regression/ and the
   training metadata to outputs/model_results/.

The test split is NOT touched here. It is used only once, in
src/model_comparison.py, after the threshold has been chosen on validation.

Why Logistic Regression for this project
----------------------------------------
It is a linear model: fast to train on millions of rows, and its coefficients
can be read directly ("a larger drop in the sender's balance raises the fraud
probability"), which makes it easy to explain in a viva. It is the natural
baseline against which a more flexible model like Random Forest is judged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.storagelevel import StorageLevel

import config
import ml_preprocessing as mlp
import model_evaluation as mev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Logistic Regression for FraudShield")
    p.add_argument("--sample", type=float, default=None,
                   help="Train on a random fraction of the data (debug only)")
    p.add_argument("--no-tune", action="store_true",
                   help="Skip the regularisation search and use regParam=0.01")
    p.add_argument("--no-mismatch", action="store_true",
                   help="Exclude the two balance-mismatch flags (ablation run)")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--model-dir", type=str, default=None,
                   help="Override where the model is saved")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = config.setup_logging()
    config.ensure_dirs()

    include_mismatch = not args.no_mismatch
    model_dir = args.model_dir or str(config.LR_MODEL_DIR)
    tag = "logistic_regression" + ("" if include_mismatch else "_no_mismatch")

    spark = config.get_spark("FraudShield_Train_LogisticRegression")
    started = time.time()

    try:
        # ---------------- data ----------------
        train, val, test = mlp.load_splits(spark, sample_fraction=args.sample)

        n_train = train.count()
        n_val = val.count()
        log.info("Train rows=%s  Validation rows=%s", f"{n_train:,}", f"{n_val:,}")

        w_legit, w_fraud = mlp.compute_class_weights(train)
        train = mlp.add_class_weight_column(train, w_legit, w_fraud)

        # ---------------- feature pipeline ----------------
        # scale=True: Logistic Regression is sensitive to feature scale.
        pipeline, features_col, numeric_cols = mlp.build_feature_pipeline(
            include_mismatch=include_mismatch, scale=True
        )
        log.info("Fitting feature pipeline ...")
        feature_model = pipeline.fit(train)

        train_feat = feature_model.transform(train).select(
            features_col, config.TARGET_COL, mlp.WEIGHT_COL
        )
        val_feat = feature_model.transform(val).select(features_col, config.TARGET_COL)

        # The grid search reads the training features repeatedly, so materialise
        # them once. MEMORY_AND_DISK spills instead of failing on an 8 GB laptop.
        train_feat.persist(StorageLevel.MEMORY_AND_DISK)
        val_feat.persist(StorageLevel.MEMORY_AND_DISK)
        log.info("Materialised training features: %s rows", f"{train_feat.count():,}")

        # ---------------- hyperparameter search on VALIDATION ----------------
        # A plain loop over regParam, scored on the chronological validation
        # split. Spark's CrossValidator is deliberately NOT used: k-fold CV
        # shuffles rows across time and would leak future transactions into
        # training, which is exactly what the chronological split prevents.
        reg_params = [0.01] if args.no_tune else [0.0, 0.001, 0.01, 0.1]
        candidates = []

        for reg in reg_params:
            lr = LogisticRegression(
                featuresCol=features_col,
                labelCol=config.TARGET_COL,
                weightCol=mlp.WEIGHT_COL,
                maxIter=args.max_iter,
                regParam=reg,
                elasticNetParam=0.0,   # pure L2 (ridge) regularisation
                standardization=False, # already scaled in the pipeline
            )
            t0 = time.time()
            log.info("Fitting Logistic Regression (regParam=%s) ...", reg)
            model = lr.fit(train_feat)
            fit_seconds = time.time() - t0

            preds = model.transform(val_feat)
            areas = mev.area_metrics(preds)
            try:
                iterations = model.summary.totalIterations
            except Exception:
                iterations = -1
            log.info(
                "regParam=%-6s  validation PR-AUC=%.6f  ROC-AUC=%.6f  (%.1f s, %d iterations)",
                reg, areas["pr_auc"], areas["roc_auc"], fit_seconds, iterations,
            )
            candidates.append(
                {"reg_param": reg, "fit_seconds": round(fit_seconds, 1), **areas,
                 "_model": model}
            )

        best = max(candidates, key=lambda c: c["pr_auc"])
        best_model = best.pop("_model")
        for c in candidates:
            c.pop("_model", None)
        log.info("Best regParam = %s (validation PR-AUC=%.6f)",
                 best["reg_param"], best["pr_auc"])

        # ---------------- validation metrics at the default 0.5 threshold ----
        # Reported only as a reference point. The threshold that FraudShield
        # actually uses is chosen in src/threshold_optimization.py.
        val_preds = best_model.transform(val_feat)
        val_metrics = mev.evaluate(val_preds, threshold=0.5)
        mev.log_confusion_matrix(
            val_metrics, "Logistic Regression - validation @ threshold 0.5"
        )

        # ---------------- coefficients ----------------
        coefs = list(best_model.coefficients.toArray())
        names = mlp.feature_names(feature_model, numeric_cols, vector_size=len(coefs))
        coefficients = sorted(
            [{"feature": n, "coefficient": round(float(c), 6)}
             for n, c in zip(names, coefs)],
            key=lambda d: abs(d["coefficient"]),
            reverse=True,
        )
        log.info("Top coefficients by magnitude (on scaled features):")
        for d in coefficients[:8]:
            log.info("   %-32s %+.4f", d["feature"], d["coefficient"])

        # ---------------- save ----------------
        # The feature pipeline and the classifier are saved as ONE PipelineModel
        # so that prediction only needs this single artefact.
        full_model = PipelineModel(stages=list(feature_model.stages) + [best_model])
        log.info("Saving model pipeline to %s", model_dir)
        full_model.write().overwrite().save(model_dir)

        summary = {
            "model": "LogisticRegression",
            "tag": tag,
            "spark_version": spark.version,
            "trained_on": "sample" if args.sample else "full_dataset",
            "sample_fraction": args.sample,
            "include_mismatch_features": include_mismatch,
            "numeric_features": numeric_cols,
            "feature_names": names,
            "train_rows": n_train,
            "validation_rows": n_val,
            "class_weights": {"legitimate": round(w_legit, 6), "fraud": round(w_fraud, 4)},
            "max_iter": args.max_iter,
            "candidates": candidates,
            "best_reg_param": best["reg_param"],
            "validation_pr_auc": best["pr_auc"],
            "validation_roc_auc": best["roc_auc"],
            "validation_metrics_at_0.5": val_metrics,
            "coefficients": coefficients,
            "intercept": round(float(best_model.intercept), 6),
            "model_path": model_dir,
            "total_seconds": round(time.time() - started, 1),
        }
        mev.save_json(summary, config.MODEL_RESULTS_DIR / f"{tag}_training.json")

        log.info("Logistic Regression training finished in %.1f s", time.time() - started)
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
