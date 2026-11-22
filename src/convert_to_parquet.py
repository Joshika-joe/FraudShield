"""
FraudShield - convert Member 1's engineered CSV output to Parquet.

Run once:  python src/convert_to_parquet.py

Why this step exists
--------------------
Member 1 saved the engineered dataset as 16 CSV part files (557 MB). CSV is a
plain text format: every time Spark reads it, it must re-parse 6.36 million
lines of text and convert each field to a number. Parquet is a columnar binary
format that stores the data types, compresses each column, and lets Spark read
only the columns a job actually needs.

For this project that matters because the ML scripts read the dataset many
times (training, validation, threshold tuning, test evaluation, dashboard
aggregates). Converting once makes every later read several times faster and
removes any risk of CSV type-guessing mistakes.

This also satisfies the project requirement to demonstrate Parquet-based
persistence in a Big Data workflow. Member 1's CSV output is left untouched.

A `row_id` column is added here as a stable display identifier for fraud
alerts. PaySim has no transaction ID, and the account name columns were
(correctly) dropped during preprocessing, so alerts need some safe way to point
at a specific transaction. `row_id` is NEVER used as a model input.
"""

from __future__ import annotations

import sys
import time

from pyspark.sql import functions as F

import config


def main() -> int:
    log = config.setup_logging()
    config.ensure_dirs()

    spark = config.get_spark("FraudShield_CSV_to_Parquet")
    started = time.time()

    try:
        log.info("Reading engineered CSV directory: %s", config.ENGINEERED_CSV_DIR)
        df = config.read_engineered(spark, prefer_parquet=False)

        log.info("Input columns (%d): %s", len(df.columns), ", ".join(df.columns))

        # monotonically_increasing_id() is a Spark function that assigns unique
        # 64-bit ids without a shuffle. The value is written into the Parquet
        # file, so it stays the same for every later read.
        df = df.withColumn("row_id", F.monotonically_increasing_id())

        # Keep the same number of output files as the input (16), which suits
        # this 8-core machine: enough parallelism, no tiny-file overhead.
        log.info("Writing Parquet to: %s", config.ENGINEERED_PARQUET_DIR)
        (
            df.write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(str(config.ENGINEERED_PARQUET_DIR))
        )

        # --- verify the conversion lost nothing ---
        parquet_df = spark.read.parquet(str(config.ENGINEERED_PARQUET_DIR))
        rows = parquet_df.count()
        fraud = parquet_df.filter(F.col(config.TARGET_COL) == 1).count()
        distinct_ids = parquet_df.select("row_id").distinct().count()

        log.info("Parquet rows        : %s", f"{rows:,}")
        log.info("Parquet fraud rows  : %s", f"{fraud:,}")
        log.info("Distinct row_id     : %s", f"{distinct_ids:,}")

        ok = (
            rows == config.EXPECTED_ROWS
            and fraud == config.EXPECTED_FRAUD
            and distinct_ids == rows
        )
        if not ok:
            log.error(
                "Conversion mismatch! expected rows=%s fraud=%s unique ids=rows",
                f"{config.EXPECTED_ROWS:,}",
                f"{config.EXPECTED_FRAUD:,}",
            )
            return 1

        log.info(
            "Conversion verified successfully in %.1f s", time.time() - started
        )
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
