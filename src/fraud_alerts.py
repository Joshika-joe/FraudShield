"""
FraudShield - Module 8b: fraud alert generation.

Usage
-----
    python src/fraud_alerts.py                       # score the test split
    python src/fraud_alerts.py --source all          # score all 6.36 M rows
    python src/fraud_alerts.py --model lr            # use a specific model
    python src/fraud_alerts.py --input path/to/data  # score a new CSV/Parquet

IMPORTANT - what this is
------------------------
This is a BATCH fraud detection demonstration. It reads a file of completed
transactions, scores them with the saved model, and writes a list of
suspicious ones. It is NOT connected to a bank, a payment switch or any live
transaction feed, and it does not block or reverse anything. In a real
deployment the same saved pipeline would sit behind a streaming job or a
service call; the scoring logic would be identical, only the input would
change.

How an alert is produced
------------------------
1. Load the saved Spark MLlib pipeline (feature transformations + classifier).
2. Score every transaction to get a fraud probability.
3. Flag a transaction when its probability reaches the operating threshold
   that was chosen on the validation split (NOT 0.5).
4. Band each flagged transaction into High / Medium / Low Risk using the
   probability bands derived from validation precision.
5. Write the alerts to Parquet (full, for Spark) and CSV (for the dashboard
   and for download).

Outputs (in outputs/fraud_alerts/)
    fraud_alerts.parquet        every alert, Spark-native
    fraud_alerts.csv            alerts for the dashboard, highest risk first
    fraud_alerts_summary.json   counts, totals and the policy that was applied
"""

from __future__ import annotations

import argparse
import sys
import time

from pyspark.sql import functions as F

import config
import model_evaluation as mev
import model_prediction as mp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate FraudShield fraud alerts")
    p.add_argument("--model", choices=["final", "lr", "rf"], default="final",
                   help="Which saved model to use ('final' = chosen on validation)")
    p.add_argument("--source", choices=["test", "validation", "train", "all"],
                   default="test",
                   help="Which part of the engineered dataset to score")
    p.add_argument("--input", type=str, default=None,
                   help="Score an external CSV or Parquet file instead")
    p.add_argument("--csv-limit", type=int, default=100_000,
                   help="Maximum rows written to the CSV export (Parquet keeps all)")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def load_input(spark, args, log):
    """Pick the transactions to score."""
    if args.input:
        path = args.input
        log.info("Reading external input: %s", path)
        if path.endswith(".csv") or (path.rstrip("/").endswith("_data")):
            df = spark.read.csv(path, header=True, schema=config.ENGINEERED_SCHEMA)
        else:
            df = spark.read.parquet(path)
        return df, f"external:{path}"

    df = config.read_engineered(spark)
    if args.source == "all":
        return df, "all_transactions"
    train, val, test = config.split_chronologically(df)
    return (
        {"train": train, "validation": val, "test": test}[args.source],
        f"{args.source}_split",
    )


def main() -> int:
    args = parse_args()
    log = config.setup_logging()
    config.ensure_dirs()
    out_dir = config.FRAUD_ALERTS_DIR if args.out_dir is None else __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = config.get_spark("FraudShield_Fraud_Alerts")
    started = time.time()

    try:
        scorer = mp.load_scorer(spark, args.model)
        log.info("Scoring with %s", scorer.describe())
        if not scorer.policy:
            log.error(
                "No tuned threshold policy was found, so alerts would use an "
                "arbitrary 0.5 cut-off. Run src/threshold_optimization.py first."
            )
            return 1

        df, source_name = load_input(spark, args, log)
        n_input = df.count()
        log.info("Transactions to score: %s (source: %s)", f"{n_input:,}", source_name)

        scored = mp.score_transactions(df, scorer).cache()
        alerts = mp.alerts_only(scored).cache()
        n_alerts = alerts.count()
        log.info(
            "Alerts raised: %s (%.4f %% of the scored transactions)",
            f"{n_alerts:,}", 100.0 * n_alerts / n_input if n_input else 0.0,
        )

        # ---------------- risk breakdown ----------------
        log.info("--- ALERTS BY RISK CATEGORY ---")
        risk_rows = (
            alerts.groupBy("risk_category")
            .agg(
                F.count("*").alias("alerts"),
                F.round(F.sum("amount"), 2).alias("total_amount"),
                F.round(F.avg(mev.FRAUD_PROB_COL), 4).alias("avg_probability"),
            )
            .collect()
        )
        risk_breakdown = []
        for r in risk_rows:
            d = r.asDict()
            d["total_amount"] = float(d["total_amount"] or 0.0)
            d["avg_probability"] = float(d["avg_probability"] or 0.0)
            risk_breakdown.append(d)
            log.info(
                "%-12s alerts=%-8s total_amount=%-18s avg_probability=%.4f",
                d["risk_category"], f"{d['alerts']:,}",
                f"{d['total_amount']:,.2f}", d["avg_probability"],
            )

        # ---------------- accuracy of the alerts, when labels exist ----------
        label_available = config.TARGET_COL in scored.columns
        alert_quality = None
        if label_available:
            counts = scored.agg(
                F.sum(F.when((F.col("predicted_class") == 1) & (F.col(config.TARGET_COL) == 1), 1).otherwise(0)).alias("tp"),
                F.sum(F.when((F.col("predicted_class") == 1) & (F.col(config.TARGET_COL) == 0), 1).otherwise(0)).alias("fp"),
                F.sum(F.when((F.col("predicted_class") == 0) & (F.col(config.TARGET_COL) == 1), 1).otherwise(0)).alias("fn"),
                F.sum(F.when((F.col("predicted_class") == 0) & (F.col(config.TARGET_COL) == 0), 1).otherwise(0)).alias("tn"),
            ).collect()[0]
            tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            alert_quality = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "fraud_caught": tp,
                "fraud_missed": fn,
            }
            log.info(
                "Against the known labels in this data: caught %s of %s fraudulent "
                "transactions (recall=%.4f), with %s false alarms (precision=%.4f)",
                f"{tp:,}", f"{tp + fn:,}", recall, f"{fp:,}", precision,
            )

        # ---------------- save ----------------
        parquet_path = out_dir / "fraud_alerts.parquet"
        log.info("Writing all alerts to %s", parquet_path)
        alerts.write.mode("overwrite").parquet(str(parquet_path))

        # Highest-probability alerts first, capped, as a single CSV file that
        # the Streamlit dashboard and the download button can read directly.
        csv_path = out_dir / "fraud_alerts.csv"
        log.info("Writing CSV export (max %s rows) to %s", f"{args.csv_limit:,}", csv_path)
        pdf = (
            alerts.orderBy(F.col(mev.FRAUD_PROB_COL).desc())
            .limit(args.csv_limit)
            .toPandas()
        )
        pdf.to_csv(csv_path, index=False)
        log.info("CSV rows written: %s", f"{len(pdf):,}")

        total_flagged_amount = float(
            alerts.agg(F.sum("amount")).collect()[0][0] or 0.0
        )

        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "batch scoring demonstration (not a live bank connection)",
            "model": scorer.name,
            "model_key": scorer.key,
            "model_path": str(scorer.model_path),
            "source": source_name,
            "transactions_scored": n_input,
            "alerts_raised": n_alerts,
            "alert_rate_pct": round(100.0 * n_alerts / n_input, 6) if n_input else 0.0,
            "total_flagged_amount": round(total_flagged_amount, 2),
            "policy": {
                "operating_threshold": scorer.threshold,
                "high_risk_threshold": scorer.high_risk_threshold,
                "medium_risk_threshold": scorer.medium_risk_threshold,
                "threshold_selection_policy": scorer.policy.get("selection_policy"),
                "risk_band_basis": scorer.policy.get("risk_bands", {}).get(
                    "basis", "derived from the validation split"
                ),
                "fallbacks": scorer.policy.get("risk_bands", {}).get("fallbacks", []),
            },
            "risk_breakdown": risk_breakdown,
            "alert_quality_vs_labels": alert_quality,
            "outputs": {
                "parquet": str(parquet_path),
                "csv": str(csv_path),
                "csv_rows": len(pdf),
                "csv_capped": len(pdf) >= args.csv_limit,
            },
            "elapsed_seconds": round(time.time() - started, 1),
        }
        mev.save_json(summary, out_dir / "fraud_alerts_summary.json")

        log.info("--- SAMPLE ALERTS (highest probability first) ---")
        alerts.orderBy(F.col(mev.FRAUD_PROB_COL).desc()).select(
            "row_id", "step", "type", "amount", mev.FRAUD_PROB_COL,
            "predicted_class", "risk_category",
        ).show(10, truncate=False)

        log.info("Fraud alert generation finished in %.1f s", time.time() - started)
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
