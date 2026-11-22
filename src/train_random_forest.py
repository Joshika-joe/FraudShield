"""
FraudShield - Model 2: Random Forest Classifier (Spark MLlib).

Usage
-----
    python src/train_random_forest.py                 # full dataset, small grid
    python src/train_random_forest.py --sample 0.02   # fast debug run
    python src/train_random_forest.py --no-tune       # one fit, default settings
    python src/train_random_forest.py --no-mismatch   # ablation run

What this script does
---------------------
Same workflow as the Logistic Regression script - chronological split, class
weights from the training split only, MLlib pipeline, model selection on the
validation split's PR-AUC - with two differences:

* No scaling. A decision tree compares one feature against a threshold at a
  time, so multiplying a feature by a constant cannot change which split the
  tree chooses. Scaling would only add computation.
* The grid is over forest shape (number of trees, tree depth) instead of
  regularisation strength.

Why Random Forest for this project
----------------------------------
Fraud in PaySim is not a straight line through the data: it is a combination
of conditions (a TRANSFER or CASH_OUT, an amount close to the sender's whole
balance, a destination balance that does not move as it should). A forest of
decision trees captures those interactions automatically, where Logistic
Regression can only add up weighted features. The cost is a model that is
heavier to train and harder to read than a set of coefficients.

Resource note: this laptop has 8 GB of RAM and 8 cores, so the forest is kept
deliberately modest (<= 120 trees, depth <= 14) and each tree is grown on a
subsample of the rows. The code itself has no size limit - on a cluster the
same script would run with a larger grid by changing the lists below.
"""

from __future__ import annotations

import argparse
import sys
import time

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.storagelevel import StorageLevel

import config
import ml_preprocessing as mlp
import model_evaluation as mev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Random Forest for FraudShield")
    p.add_argument("--sample", type=float, default=None,
                   help="Train on a random fraction of the data (debug only)")
    p.add_argument("--no-tune", action="store_true",
                   help="Fit once with numTrees=80, maxDepth=10")
    p.add_argument("--no-mismatch", action="store_true",
                   help="Exclude the two balance-mismatch flags (ablation run)")
    p.add_argument("--num-trees", type=int, default=None,
                   help="Fix the number of trees (skips that part of the grid)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Fix the tree depth (skips that part of the grid)")
    p.add_argument("--subsampling-rate", type=float, default=0.6,
                   help="Fraction of training rows used to grow each tree")
    p.add_argument("--model-dir", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = config.setup_logging()
    config.ensure_dirs()

    include_mismatch = not args.no_mismatch
    model_dir = args.model_dir or str(config.RF_MODEL_DIR)
    tag = "random_forest" + ("" if include_mismatch else "_no_mismatch")

    spark = config.get_spark("FraudShield_Train_RandomForest")
    started = time.time()

    try:
        # ---------------- data ----------------
        train, val, test = mlp.load_splits(spark, sample_fraction=args.sample)
        n_train = train.count()
        n_val = val.count()
        log.info("Train rows=%s  Validation rows=%s", f"{n_train:,}", f"{n_val:,}")

        w_legit, w_fraud = mlp.compute_class_weights(train)
        train = mlp.add_class_weight_column(train, w_legit, w_fraud)

        # ---------------- feature pipeline (no scaling) ----------------
        pipeline, features_col, numeric_cols = mlp.build_feature_pipeline(
            include_mismatch=include_mismatch, scale=False
        )
        log.info("Fitting feature pipeline ...")
        feature_model = pipeline.fit(train)

        train_feat = feature_model.transform(train).select(
            features_col, config.TARGET_COL, mlp.WEIGHT_COL
        )
        val_feat = feature_model.transform(val).select(features_col, config.TARGET_COL)
        train_feat.persist(StorageLevel.MEMORY_AND_DISK)
        val_feat.persist(StorageLevel.MEMORY_AND_DISK)
        log.info("Materialised training features: %s rows", f"{train_feat.count():,}")

        # ---------------- grid over forest shape ----------------
        if args.no_tune:
            grid = [(args.num_trees or 80, args.max_depth or 10)]
        else:
            trees_options = [args.num_trees] if args.num_trees else [60, 120]
            depth_options = [args.max_depth] if args.max_depth else [8, 14]
            grid = [(t, d) for t in trees_options for d in depth_options]

        log.info("Grid: %s", grid)
        candidates = []

        for num_trees, max_depth in grid:
            rf = RandomForestClassifier(
                featuresCol=features_col,
                labelCol=config.TARGET_COL,
                weightCol=mlp.WEIGHT_COL,
                numTrees=num_trees,
                maxDepth=max_depth,
                maxBins=32,
                subsamplingRate=args.subsampling_rate,
                featureSubsetStrategy="sqrt",
                minInstancesPerNode=20,
                seed=config.RANDOM_SEED,
                cacheNodeIds=True,
                maxMemoryInMB=512,
            )
            t0 = time.time()
            log.info("Fitting Random Forest (numTrees=%d, maxDepth=%d) ...",
                     num_trees, max_depth)
            model = rf.fit(train_feat)
            fit_seconds = time.time() - t0

            areas = mev.area_metrics(model.transform(val_feat))
            log.info(
                "numTrees=%-4d maxDepth=%-3d  validation PR-AUC=%.6f  ROC-AUC=%.6f  (%.1f s)",
                num_trees, max_depth, areas["pr_auc"], areas["roc_auc"], fit_seconds,
            )
            candidates.append(
                {"num_trees": num_trees, "max_depth": max_depth,
                 "fit_seconds": round(fit_seconds, 1), **areas, "_model": model}
            )

        best = max(candidates, key=lambda c: c["pr_auc"])
        best_model = best.pop("_model")
        for c in candidates:
            c.pop("_model", None)
        log.info("Best forest: numTrees=%d maxDepth=%d (validation PR-AUC=%.6f)",
                 best["num_trees"], best["max_depth"], best["pr_auc"])

        # ---------------- validation metrics at 0.5 (reference only) ----------
        val_metrics = mev.evaluate(best_model.transform(val_feat), threshold=0.5)
        mev.log_confusion_matrix(val_metrics, "Random Forest - validation @ threshold 0.5")

        # ---------------- feature importances ----------------
        importance_values = list(best_model.featureImportances.toArray())
        names = mlp.feature_names(
            feature_model, numeric_cols, vector_size=len(importance_values)
        )
        importances = sorted(
            [{"feature": n, "importance": round(float(v), 6)}
             for n, v in zip(names, importance_values)],
            key=lambda d: d["importance"],
            reverse=True,
        )
        log.info("Feature importances (how much each feature reduced impurity):")
        for d in importances[:8]:
            log.info("   %-32s %.4f", d["feature"], d["importance"])

        # ---------------- save ----------------
        full_model = PipelineModel(stages=list(feature_model.stages) + [best_model])
        log.info("Saving model pipeline to %s", model_dir)
        full_model.write().overwrite().save(model_dir)

        summary = {
            "model": "RandomForestClassifier",
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
            "subsampling_rate": args.subsampling_rate,
            "feature_subset_strategy": "sqrt",
            "candidates": candidates,
            "best_num_trees": best["num_trees"],
            "best_max_depth": best["max_depth"],
            "validation_pr_auc": best["pr_auc"],
            "validation_roc_auc": best["roc_auc"],
            "validation_metrics_at_0.5": val_metrics,
            "feature_importances": importances,
            "total_nodes": int(best_model.totalNumNodes),
            "model_path": model_dir,
            "total_seconds": round(time.time() - started, 1),
        }
        mev.save_json(summary, config.MODEL_RESULTS_DIR / f"{tag}_training.json")

        log.info("Random Forest training finished in %.1f s", time.time() - started)
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
