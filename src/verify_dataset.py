"""
FraudShield - verify Member 1's engineered dataset with Spark (Phase 2).

Run:  python src/verify_dataset.py

What it checks
--------------
1. The dataset can be read by Spark (CSV directory or Parquet copy).
2. Schema: 13 engineered columns with the expected types.
3. Row count, fraud count, legitimate count.
4. Missing values in every column.
5. The five engineered features exist and are internally consistent
   (recomputed from the raw balance columns and compared).
6. Transaction type distribution and fraud-by-type (Spark SQL).
7. The chronological train / validation / test split actually produced
   enough fraud examples in each window.

Everything is computed with Spark DataFrame operations and Spark SQL on the
full 6.36 million rows, not on a sample. Results are written to
outputs/model_results/dataset_verification.json so the dashboard and the
report can quote measured numbers instead of hard-coded ones.
"""

from __future__ import annotations

import json
import sys
import time
from decimal import Decimal

from pyspark.sql import functions as F

import config


def main() -> int:
    log = config.setup_logging()
    config.ensure_dirs()
    spark = config.get_spark("FraudShield_Dataset_Verification")
    started = time.time()
    report: dict = {}

    try:
        df = config.read_engineered(spark)
        source = (
            "parquet"
            if (config.ENGINEERED_PARQUET_DIR / "_SUCCESS").exists()
            else "csv"
        )
        report["source_format"] = source
        log.info("Source format: %s", source)

        # Cache: this script scans the data several times.
        df.cache()

        # ---------------- 1. schema ----------------
        log.info("--- SCHEMA ---")
        df.printSchema()
        report["columns"] = df.columns
        report["n_columns"] = len(df.columns)

        expected_cols = [f.name for f in config.ENGINEERED_SCHEMA.fields]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        report["missing_expected_columns"] = missing_cols
        if missing_cols:
            log.error("Missing expected columns: %s", missing_cols)

        # ---------------- 2. counts ----------------
        log.info("--- ROW COUNT AND TARGET DISTRIBUTION ---")
        total = df.count()
        fraud = df.filter(F.col(config.TARGET_COL) == 1).count()
        legit = total - fraud
        report.update(
            total_rows=total,
            fraud_rows=fraud,
            legitimate_rows=legit,
            fraud_percentage=round(100.0 * fraud / total, 6),
        )
        log.info("Total rows        : %s", f"{total:,}")
        log.info("Fraud rows        : %s", f"{fraud:,}")
        log.info("Legitimate rows   : %s", f"{legit:,}")
        log.info("Fraud percentage  : %.4f %%", 100.0 * fraud / total)

        checks = {
            "row_count_matches_expected": total == config.EXPECTED_ROWS,
            "fraud_count_matches_expected": fraud == config.EXPECTED_FRAUD,
            "legit_count_matches_expected": legit == config.EXPECTED_LEGIT,
        }

        # ---------------- 3. missing values ----------------
        log.info("--- MISSING VALUE CHECK ---")
        null_row = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
        ).collect()[0]
        nulls = {c: int(null_row[c]) for c in df.columns}
        report["null_counts"] = nulls
        total_nulls = sum(nulls.values())
        checks["no_missing_values"] = total_nulls == 0
        log.info("Total missing values across all columns: %d", total_nulls)
        if total_nulls:
            log.warning(
                "Columns with nulls: %s", {k: v for k, v in nulls.items() if v}
            )

        # ---------------- 4. engineered feature consistency ----------------
        # Recompute each engineered feature from the raw columns and count how
        # many rows disagree with the stored value. This proves the saved
        # features really are what README_ML_HANDOVER.md says they are.
        log.info("--- ENGINEERED FEATURE CONSISTENCY ---")
        tol = 0.011  # the CSV stores values rounded to 2 decimals
        recomputed = df.select(
            F.abs(
                F.col("sender_balance_change")
                - (F.col("oldbalanceOrg") - F.col("newbalanceOrig"))
            ).alias("d_sender"),
            F.abs(
                F.col("destination_balance_change")
                - (F.col("newbalanceDest") - F.col("oldbalanceDest"))
            ).alias("d_dest"),
            F.abs(F.col("log_amount") - F.log1p(F.col("amount"))).alias("d_log"),
            (
                F.col("sender_balance_mismatch")
                != F.when(
                    F.abs(
                        F.col("oldbalanceOrg")
                        - F.col("amount")
                        - F.col("newbalanceOrig")
                    )
                    > 0.01,
                    1,
                ).otherwise(0)
            ).alias("bad_sender_flag"),
            (
                F.col("destination_balance_mismatch")
                != F.when(
                    F.abs(
                        F.col("oldbalanceDest")
                        + F.col("amount")
                        - F.col("newbalanceDest")
                    )
                    > 0.01,
                    1,
                ).otherwise(0)
            ).alias("bad_dest_flag"),
        )
        agg = recomputed.select(
            F.sum(F.when(F.col("d_sender") > tol, 1).otherwise(0)).alias("sender_bad"),
            F.sum(F.when(F.col("d_dest") > tol, 1).otherwise(0)).alias("dest_bad"),
            F.sum(F.when(F.col("d_log") > 1e-9, 1).otherwise(0)).alias("log_bad"),
            F.sum(F.col("bad_sender_flag").cast("int")).alias("sender_flag_bad"),
            F.sum(F.col("bad_dest_flag").cast("int")).alias("dest_flag_bad"),
        ).collect()[0]

        feature_mismatches = {
            "sender_balance_change": int(agg["sender_bad"]),
            "destination_balance_change": int(agg["dest_bad"]),
            "log_amount": int(agg["log_bad"]),
            "sender_balance_mismatch": int(agg["sender_flag_bad"]),
            "destination_balance_mismatch": int(agg["dest_flag_bad"]),
        }
        report["feature_recompute_mismatches"] = feature_mismatches
        for name, bad in feature_mismatches.items():
            log.info("%-30s rows disagreeing with formula: %d", name, bad)
        checks["engineered_features_consistent"] = all(
            v == 0 for v in feature_mismatches.values()
        )

        # Base rate of the two mismatch flags (they turned out to be very common).
        flag_rates = df.select(
            F.avg("sender_balance_mismatch").alias("sender_rate"),
            F.avg("destination_balance_mismatch").alias("dest_rate"),
        ).collect()[0]
        report["sender_mismatch_rate_pct"] = round(100 * flag_rates["sender_rate"], 4)
        report["destination_mismatch_rate_pct"] = round(100 * flag_rates["dest_rate"], 4)
        log.info(
            "sender_balance_mismatch = 1 in %.2f %% of rows; "
            "destination_balance_mismatch = 1 in %.2f %% of rows",
            100 * flag_rates["sender_rate"],
            100 * flag_rates["dest_rate"],
        )

        # ---------------- 5. transaction types (Spark SQL) ----------------
        log.info("--- FRAUD BY TRANSACTION TYPE (Spark SQL) ---")
        df.createOrReplaceTempView("transactions")
        type_rows = spark.sql(
            """
            SELECT  type,
                    COUNT(*)                                  AS total,
                    SUM(isFraud)                              AS fraud,
                    ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_pct,
                    ROUND(AVG(amount), 2)                     AS avg_amount
            FROM    transactions
            GROUP BY type
            ORDER BY fraud DESC
            """
        ).collect()
        report["by_type"] = [r.asDict() for r in type_rows]
        for r in type_rows:
            log.info(
                "%-10s total=%-9s fraud=%-6s fraud_pct=%-8s avg_amount=%s",
                r["type"],
                f"{r['total']:,}",
                f"{r['fraud']:,}",
                r["fraud_pct"],
                f"{r['avg_amount']:,.2f}",
            )

        # ---------------- 6. chronological split ----------------
        log.info("--- CHRONOLOGICAL SPLIT (step boundaries %d / %d) ---",
                 config.TRAIN_MAX_STEP, config.VAL_MAX_STEP)
        split_rows = (
            config.add_split_column(df)
            .groupBy("split")
            .agg(
                F.count("*").alias("rows"),
                F.sum(config.TARGET_COL).alias("fraud"),
                F.min("step").alias("min_step"),
                F.max("step").alias("max_step"),
            )
            .collect()
        )
        split_info = {}
        for r in split_rows:
            d = r.asDict()
            d["rows_pct"] = round(100.0 * d["rows"] / total, 2)
            d["fraud_pct"] = round(100.0 * d["fraud"] / d["rows"], 4)
            split_info[d.pop("split")] = d
        report["splits"] = split_info
        for name in ("train", "validation", "test"):
            d = split_info[name]
            log.info(
                "%-10s steps %3d-%3d  rows=%-10s (%5.2f%%)  fraud=%-6s (%.4f%%)",
                name, d["min_step"], d["max_step"], f"{d['rows']:,}",
                d["rows_pct"], f"{d['fraud']:,}", d["fraud_pct"],
            )

        # Every split needs enough fraud rows to measure recall meaningfully.
        checks["val_has_enough_fraud"] = split_info["validation"]["fraud"] >= 100
        checks["test_has_enough_fraud"] = split_info["test"]["fraud"] >= 100
        checks["splits_are_chronological"] = (
            split_info["train"]["max_step"] < split_info["validation"]["min_step"]
            < split_info["validation"]["max_step"] < split_info["test"]["min_step"]
        )

        # ---------------- summary ----------------
        report["checks"] = checks
        report["all_checks_passed"] = all(checks.values())
        report["elapsed_seconds"] = round(time.time() - started, 1)

        # Spark SQL's ROUND() returns java.math.BigDecimal, which arrives in
        # Python as decimal.Decimal and is not JSON serialisable by default.
        def _jsonable(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            return str(obj)

        out = config.MODEL_RESULTS_DIR / "dataset_verification.json"
        out.write_text(json.dumps(report, indent=2, default=_jsonable))

        log.info("--- VERIFICATION SUMMARY ---")
        for name, passed in checks.items():
            log.info("%-38s %s", name, "PASS" if passed else "FAIL")
        log.info("Report written to %s", out)
        log.info("Completed in %.1f s", time.time() - started)

        return 0 if report["all_checks_passed"] else 1

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
