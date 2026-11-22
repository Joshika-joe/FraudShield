"""
FraudShield - precompute the aggregates the Streamlit dashboard displays.

Usage
-----
    python src/prepare_dashboard_data.py

Why this script exists
----------------------
Streamlit re-runs its whole script on every click. If the dashboard read the
6.36 million row dataset each time, every interaction would take minutes and
would need a running Spark session behind the web app.

So the heavy work happens HERE, once, in Spark: all of the analytics are
computed on the full dataset and written out as small CSV/JSON summaries (a
few hundred kilobytes in total). The dashboard then only reads those summaries
with pandas and renders them, which is instant.

This is the standard Big Data pattern: distributed engine for computation,
precomputed aggregates for serving.

Outputs (in outputs/dashboard_data/)
    overview.json             headline KPIs
    type_distribution.csv     per transaction type: volume, fraud, amounts
    fraud_by_step.csv         fraud count and rate for each simulation step
    hourly_pattern.csv        fraud by hour-of-day (step modulo 24)
    amount_bands.csv          transaction counts by amount band, fraud vs legit
    amount_stats.csv          amount statistics for fraud vs legitimate
    mismatch_stats.csv        balance-mismatch flag rates, fraud vs legitimate
    split_summary.csv         rows and fraud per train/validation/test window
    sample_transactions.csv   small stratified sample for interactive browsing
"""

from __future__ import annotations

import sys
import time

from pyspark.sql import functions as F

import config
import model_evaluation as mev


def write_csv(pdf, path, log) -> None:
    pdf.to_csv(path, index=False)
    log.info("Saved %-28s (%d rows)", path.name, len(pdf))


def main() -> int:
    log = config.setup_logging()
    config.ensure_dirs()
    out = config.DASHBOARD_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    spark = config.get_spark("FraudShield_Dashboard_Data")
    started = time.time()

    try:
        df = config.read_engineered(spark)
        df = config.add_split_column(df).cache()
        df.createOrReplaceTempView("transactions")

        total = df.count()
        log.info("Computing dashboard aggregates over %s transactions", f"{total:,}")

        # ---------------- 1. overview KPIs ----------------
        row = df.agg(
            F.sum(config.TARGET_COL).alias("fraud"),
            F.sum(F.when(F.col(config.TARGET_COL) == 1, F.col("amount"))).alias("fraud_amount"),
            F.avg(F.when(F.col(config.TARGET_COL) == 1, F.col("amount"))).alias("avg_fraud_amount"),
            F.max(F.when(F.col(config.TARGET_COL) == 1, F.col("amount"))).alias("max_fraud_amount"),
            F.sum("amount").alias("total_amount"),
            F.avg("amount").alias("avg_amount"),
            F.min("step").alias("min_step"),
            F.max("step").alias("max_step"),
        ).collect()[0]

        fraud = int(row["fraud"])
        overview = {
            "total_transactions": total,
            "fraud_transactions": fraud,
            "legitimate_transactions": total - fraud,
            "fraud_percentage": round(100.0 * fraud / total, 4),
            "legitimate_percentage": round(100.0 * (total - fraud) / total, 4),
            "total_fraud_amount": round(float(row["fraud_amount"]), 2),
            "average_fraud_amount": round(float(row["avg_fraud_amount"]), 2),
            "maximum_fraud_amount": round(float(row["max_fraud_amount"]), 2),
            "total_transaction_amount": round(float(row["total_amount"]), 2),
            "average_transaction_amount": round(float(row["avg_amount"]), 2),
            "min_step": int(row["min_step"]),
            "max_step": int(row["max_step"]),
            "simulated_days": round(int(row["max_step"]) / 24.0, 1),
            "dataset": "PaySim synthetic financial transactions (engineered by Member 1)",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        mev.save_json(overview, out / "overview.json")

        # ---------------- 2. transaction type distribution ----------------
        type_pdf = spark.sql(
            """
            SELECT type,
                   COUNT(*)                                   AS transactions,
                   SUM(isFraud)                               AS fraud,
                   COUNT(*) - SUM(isFraud)                    AS legitimate,
                   ROUND(100.0 * SUM(isFraud) / COUNT(*), 4)  AS fraud_pct,
                   ROUND(SUM(amount), 2)                      AS total_amount,
                   ROUND(AVG(amount), 2)                      AS avg_amount,
                   ROUND(MAX(amount), 2)                      AS max_amount,
                   ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
            FROM   transactions
            GROUP BY type
            ORDER BY transactions DESC
            """
        ).toPandas()
        write_csv(type_pdf, out / "type_distribution.csv", log)

        # ---------------- 3. fraud trend by step ----------------
        step_pdf = spark.sql(
            """
            SELECT step,
                   COUNT(*)                                  AS transactions,
                   SUM(isFraud)                              AS fraud,
                   ROUND(100.0 * SUM(isFraud) / COUNT(*), 6) AS fraud_pct,
                   ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
            FROM   transactions
            GROUP BY step
            ORDER BY step
            """
        ).toPandas()
        write_csv(step_pdf, out / "fraud_by_step.csv", log)

        # ---------------- 4. hour-of-day pattern ----------------
        # PaySim's step is one simulated hour, so step % 24 behaves like an
        # hour of the day. This is a simulation artefact, not real clock time,
        # and the dashboard says so.
        hour_pdf = spark.sql(
            """
            SELECT step % 24                                 AS hour_of_day,
                   COUNT(*)                                  AS transactions,
                   SUM(isFraud)                              AS fraud,
                   ROUND(100.0 * SUM(isFraud) / COUNT(*), 6) AS fraud_pct
            FROM   transactions
            GROUP BY step % 24
            ORDER BY hour_of_day
            """
        ).toPandas()
        write_csv(hour_pdf, out / "hourly_pattern.csv", log)

        # ---------------- 5. amount bands ----------------
        banded = df.withColumn(
            "amount_band",
            F.when(F.col("amount") < 1_000, "0 - 1K")
            .when(F.col("amount") < 10_000, "1K - 10K")
            .when(F.col("amount") < 50_000, "10K - 50K")
            .when(F.col("amount") < 100_000, "50K - 100K")
            .when(F.col("amount") < 500_000, "100K - 500K")
            .when(F.col("amount") < 1_000_000, "500K - 1M")
            .otherwise("1M +"),
        ).withColumn(
            "band_order",
            F.when(F.col("amount") < 1_000, 1)
            .when(F.col("amount") < 10_000, 2)
            .when(F.col("amount") < 50_000, 3)
            .when(F.col("amount") < 100_000, 4)
            .when(F.col("amount") < 500_000, 5)
            .when(F.col("amount") < 1_000_000, 6)
            .otherwise(7),
        )
        band_pdf = (
            banded.groupBy("amount_band", "band_order")
            .agg(
                F.count("*").alias("transactions"),
                F.sum(config.TARGET_COL).alias("fraud"),
                (F.count("*") - F.sum(config.TARGET_COL)).alias("legitimate"),
                F.round(100.0 * F.sum(config.TARGET_COL) / F.count("*"), 6).alias("fraud_pct"),
            )
            .orderBy("band_order")
            .toPandas()
        )
        write_csv(band_pdf, out / "amount_bands.csv", log)

        # ---------------- 6. amount statistics, fraud vs legitimate ----------
        stats_pdf = (
            df.groupBy(config.TARGET_COL)
            .agg(
                F.count("*").alias("transactions"),
                F.round(F.avg("amount"), 2).alias("avg_amount"),
                F.round(F.expr("percentile_approx(amount, 0.5)"), 2).alias("median_amount"),
                F.round(F.min("amount"), 2).alias("min_amount"),
                F.round(F.max("amount"), 2).alias("max_amount"),
                F.round(F.stddev("amount"), 2).alias("stddev_amount"),
                F.round(F.avg("oldbalanceOrg"), 2).alias("avg_sender_balance_before"),
                F.round(F.avg("newbalanceOrig"), 2).alias("avg_sender_balance_after"),
            )
            .orderBy(config.TARGET_COL)
            .toPandas()
        )
        stats_pdf["label"] = stats_pdf[config.TARGET_COL].map({0: "Legitimate", 1: "Fraud"})
        write_csv(stats_pdf, out / "amount_stats.csv", log)

        # ---------------- 7. balance mismatch behaviour ----------------
        mismatch_pdf = (
            df.groupBy(config.TARGET_COL)
            .agg(
                F.count("*").alias("transactions"),
                F.round(100.0 * F.avg("sender_balance_mismatch"), 4).alias("sender_mismatch_pct"),
                F.round(100.0 * F.avg("destination_balance_mismatch"), 4).alias("destination_mismatch_pct"),
                F.round(F.avg("sender_balance_change"), 2).alias("avg_sender_balance_change"),
                F.round(F.avg("destination_balance_change"), 2).alias("avg_destination_balance_change"),
                F.round(100.0 * F.avg(F.when(F.col("newbalanceOrig") == 0, 1).otherwise(0)), 4).alias("drained_account_pct"),
            )
            .orderBy(config.TARGET_COL)
            .toPandas()
        )
        mismatch_pdf["label"] = mismatch_pdf[config.TARGET_COL].map({0: "Legitimate", 1: "Fraud"})
        write_csv(mismatch_pdf, out / "mismatch_stats.csv", log)

        # ---------------- 8. split summary ----------------
        split_pdf = (
            df.groupBy("split")
            .agg(
                F.count("*").alias("rows"),
                F.sum(config.TARGET_COL).alias("fraud"),
                F.min("step").alias("min_step"),
                F.max("step").alias("max_step"),
                F.round(100.0 * F.sum(config.TARGET_COL) / F.count("*"), 6).alias("fraud_pct"),
            )
            .toPandas()
        )
        order = {"train": 0, "validation": 1, "test": 2}
        split_pdf = split_pdf.sort_values("split", key=lambda s: s.map(order))
        split_pdf["rows_pct"] = (100.0 * split_pdf["rows"] / total).round(2)
        write_csv(split_pdf, out / "split_summary.csv", log)

        # ---------------- 9. stratified sample for interactive browsing -----
        # Every fraud row plus a small random slice of legitimate rows. This is
        # what the dashboard's transaction explorer filters on, so that
        # interaction stays instant without loading millions of rows.
        fraud_rows = df.filter(F.col(config.TARGET_COL) == 1)
        legit_sample = df.filter(F.col(config.TARGET_COL) == 0).sample(
            withReplacement=False, fraction=0.004, seed=config.RANDOM_SEED
        )
        sample_cols = [
            "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "sender_balance_change",
            "destination_balance_change", "log_amount",
            "sender_balance_mismatch", "destination_balance_mismatch",
            config.TARGET_COL, "split",
        ]
        if "row_id" in df.columns:
            sample_cols.insert(0, "row_id")
        sample_pdf = (
            fraud_rows.select(*sample_cols)
            .union(legit_sample.select(*sample_cols))
            .toPandas()
        )
        write_csv(sample_pdf, out / "sample_transactions.csv", log)
        log.info(
            "Sample contains %s fraud and %s legitimate transactions",
            f"{int(sample_pdf[config.TARGET_COL].sum()):,}",
            f"{int((sample_pdf[config.TARGET_COL] == 0).sum()):,}",
        )

        log.info("Dashboard data prepared in %.1f s -> %s", time.time() - started, out)
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
